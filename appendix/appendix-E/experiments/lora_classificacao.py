# Appendix E - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM

# Fine-tuning do GPT-2 124M para classificar spam/ham com LoRA (Apendice E),
# no mesmo dataset e na mesma divisao do capitulo 6, para comparar as duas
# estrategias.

#   Cap. 6 : treina o ultimo bloco + LayerNorm final + cabeca (4,35% dos parametros)
#   Ap. E  : congela TUDO e treina so as matrizes A e B do LoRA (2.666.528 parametros)

# Requer que `carregar_pesos_pretrained.py` (cap. 6-7) ja tenha rodado e gerado
#   data/gpt2_pretrained.pth

# Uso (a partir da raiz do repositorio):
#    python Apendices/appendix-E/experiments/lora_classificacao.py
#    python Apendices/appendix-E/experiments/lora_classificacao.py --rank 8 --alpha 8

# Saidas:
#    experiments/resultados/lora_resultados.json   (lora_resultados_r<rank>.json se rank != 16)
#    models/appendix-E/lora_spam_r<rank>.pth        so os pesos do LoRA (fora do git)

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import tiktoken
import torch

from src.finetuning.classificador import (
    GPTClassifier,
    avaliar_classificador,
    treinar_classificador,
)
from src.finetuning.dados_cls import (
    balancear_classes,
    baixar_dataset_spam,
    carregar_spam_df,
    criar_loaders_spam,
    dividir_treino_val_teste,
)
from src.finetuning.lora import (
    LinearWithLoRA,
    contar_parametros,
    mesclar_lora,
    parametros_lora_esperados,
    preparar_classificador_lora,
    state_dict_lora,
)
from src.model.config import GPT_CONFIG_124M
from src.training.dispositivo import descrever_dispositivo, escolher_dispositivo

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT = DATA_DIR / "gpt2_pretrained.pth"
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
MODELOS_DIR = PROJECT_ROOT / "models" / "appendix-E"
RESULTADO_CAP6 = PROJECT_ROOT / "sprints" / "ch06-07" / "experiments" / "resultados" / "classificacao_resultados.json"

CFG_GPT2 = dict(GPT_CONFIG_124M)
CFG_GPT2["qkv_bias"] = True           # o GPT-2 original usa vies em Q, K e V

BATCH_SIZE = 8
MAX_LENGTH = 128                      # igual ao cap. 6


def titulo(texto):
    print()
    print("=" * 60)
    print(texto)
    print("=" * 60)


def metricas_spam(model, loader, device):
    # Precisao, revocacao e F1 da classe spam (=1), sem sklearn.
    model.eval()
    vp = fp = fn = vn = 0
    with torch.no_grad():
        for ids, rotulos in loader:
            preds = model(ids.to(device)).argmax(dim=-1).cpu()
            vp += ((preds == 1) & (rotulos == 1)).sum().item()
            fp += ((preds == 1) & (rotulos == 0)).sum().item()
            fn += ((preds == 0) & (rotulos == 1)).sum().item()
            vn += ((preds == 0) & (rotulos == 0)).sum().item()
    precisao = vp / (vp + fp) if vp + fp else 0.0
    revocacao = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
    return {"precisao": precisao, "revocacao": revocacao, "f1": f1,
            "tp": vp, "fp": fp, "fn": fn, "tn": vn}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int, default=16, help="posto do LoRA (livro: 16)")
    ap.add_argument("--alpha", type=float, default=16, help="escala do LoRA (livro: 16)")
    ap.add_argument("--lr", type=float, default=8e-4, help="taxa de aprendizado (repositorio do livro: 8e-4)")
    ap.add_argument("--epocas", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="auto", help="auto, cuda ou cpu")
    args = ap.parse_args()

    device = escolher_dispositivo(args.device)
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------
    # 1. Dataset |
    # ------------
    titulo("1. PREPARANDO O DATASET (mesma divisao do cap. 6)")
    tokenizer = tiktoken.get_encoding("gpt2")
    dados = balancear_classes(carregar_spam_df(baixar_dataset_spam(DATA_DIR)))
    treino, val, teste = dividir_treino_val_teste(dados)
    print(f"Total (balanceado): {len(dados)}  |  Treino: {len(treino)}  Validacao: {len(val)}  Teste: {len(teste)}")
    train_loader, val_loader, test_loader = criar_loaders_spam(
        treino, val, teste, tokenizer, max_length=MAX_LENGTH, batch_size=BATCH_SIZE)

    # -------------------------------------------
    # 2. Modelo: GPT-2 pre-treinado + LoRA      |
    # -------------------------------------------
    titulo("2. GPT-2 PRE-TREINADO + LoRA")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint nao encontrado: {CHECKPOINT}\n"
            "Execute primeiro: python sprints/ch06-07/experiments/carregar_pesos_pretrained.py")

    torch.manual_seed(args.seed)
    modelo = GPTClassifier(CFG_GPT2, num_classes=2)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    modelo.gpt.load_state_dict(ckpt["model_state_dict"])

    _, total_antes = contar_parametros(modelo)
    preparar_classificador_lora(modelo, rank=args.rank, alpha=args.alpha)
    modelo.to(device)
    treinaveis, total = contar_parametros(modelo)
    camadas_lora = sum(isinstance(m, LinearWithLoRA) for m in modelo.modules())
    print(f"Dispositivo           : {descrever_dispositivo(device)}")
    print(f"Camadas com LoRA      : {camadas_lora}  (rank={args.rank}, alpha={args.alpha})")
    print(f"Parametros treinaveis : {treinaveis:,}  (conta a mao: {parametros_lora_esperados(CFG_GPT2, args.rank):,})")
    print(f"Parametros do modelo  : {total_antes:,} originais + {total - total_antes:,} do LoRA")
    print(f"Fracao treinada       : {100 * treinaveis / total_antes:.2f}% dos parametros originais")

    # ------------------------------
    # 3. Antes do fine-tuning      |
    # ------------------------------
    titulo("3. BASELINE (antes do fine-tuning)")
    antes = {nome: avaliar_classificador(modelo, loader, device)
             for nome, loader in (("treino", train_loader), ("validacao", val_loader), ("teste", test_loader))}
    for nome, acc in antes.items():
        print(f"Acuracia {nome:<10}: {acc:.3f}")
    print("(~0.50: a cabeca de classificacao e aleatoria e B = 0, entao o LoRA ainda nao muda nada)")

    # ----------------
    # 4. Fine-tuning |
    # ----------------
    titulo(f"4. FINE-TUNING COM LoRA ({args.epocas} epocas, lr {args.lr})")
    optimizer = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inicio = time.time()
    historico = treinar_classificador(modelo, train_loader, val_loader, optimizer, device,
                                      num_epochs=args.epocas, eval_freq=50)
    tempo = time.time() - inicio
    memoria = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
    print(f"Tempo de treino: {tempo / 60:.2f} min" + (f" | memoria de GPU: {memoria:.2f} GB" if memoria else ""))

    # --------------------
    # 5. Avaliacao final |
    # --------------------
    titulo("5. AVALIACAO FINAL")
    depois = {nome: avaliar_classificador(modelo, loader, device)
              for nome, loader in (("treino", train_loader), ("validacao", val_loader), ("teste", test_loader))}
    for nome in depois:
        print(f"Acuracia {nome:<10}: {antes[nome]:.3f} -> {depois[nome]:.3f}")
    metr = metricas_spam(modelo, test_loader, device)
    print(f"Precisao  (teste)  : {metr['precisao']:.3f}")
    print(f"Revocacao (teste)  : {metr['revocacao']:.3f}")
    print(f"F1-score  (teste)  : {metr['f1']:.3f}")
    print(f"Matriz de confusao (teste): VP={metr['tp']} FP={metr['fp']} FN={metr['fn']} VN={metr['tn']}")

    # ---------------------------------------
    # 6. Salvar so o LoRA e mesclar pesos   |
    # ---------------------------------------
    titulo("6. SALVAR SO O LoRA E MESCLAR NOS PESOS")
    arquivo_lora = MODELOS_DIR / f"lora_spam_r{args.rank}.pth"
    torch.save(state_dict_lora(modelo), arquivo_lora)
    mb_lora = arquivo_lora.stat().st_size / 1024**2
    mb_modelo = sum(p.numel() * p.element_size() for p in modelo.parameters()) / 1024**2
    print(f"Arquivo do LoRA: {arquivo_lora.relative_to(PROJECT_ROOT)}  ({mb_lora:.1f} MB)")
    print(f"Modelo inteiro em float32: {mb_modelo:,.0f} MB  ->  o LoRA e {100 * mb_lora / mb_modelo:.1f}% disso")

    # Mesclar: W = W + (alpha/rank) * (A @ B).T -> mesmas previsoes, sem custo extra
    modelo.eval()
    ids_exemplo, _ = next(iter(test_loader))
    with torch.no_grad():
        antes_mescla = modelo(ids_exemplo.to(device))
    mesclar_lora(modelo)
    with torch.no_grad():
        depois_mescla = modelo(ids_exemplo.to(device))
    diferenca = (antes_mescla - depois_mescla).abs().max().item()
    restantes = sum(isinstance(m, LinearWithLoRA) for m in modelo.modules())
    print(f"Depois de mesclar: {restantes} camadas LoRA restantes, diferenca maxima nos logits = {diferenca:.2e}")

    # --------------------------
    # 7. Comparacao com cap. 6 |
    # --------------------------
    titulo("7. COMPARACAO COM O CAPITULO 6")
    cap6 = None
    if RESULTADO_CAP6.exists():
        cap6 = json.loads(RESULTADO_CAP6.read_text(encoding="utf-8"))
        d6 = cap6.get("depois_finetuning", {})
        print(f"{'':<22}{'Cap. 6 (ultimo bloco)':>24}{'Ap. E (LoRA)':>16}")
        print(f"{'Acuracia de teste':<22}{d6.get('acc_teste', float('nan')):>24.3f}{depois['teste']:>16.3f}")
        if "f1" in d6:
            print(f"{'F1 de teste':<22}{d6['f1']:>24.3f}{metr['f1']:>16.3f}")
    else:
        print("Resultado do cap. 6 nao encontrado. Rode sprints/ch06-07/experiments/classificacao.py")
        print("para comparar as duas estrategias.")

    resultado = {
        "config": {"rank": args.rank, "alpha": args.alpha, "lr": args.lr, "num_epochs": args.epocas,
                   "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH, "seed": args.seed,
                   "dispositivo": descrever_dispositivo(device), "torch": str(torch.__version__)},
        "parametros": {"treinaveis": treinaveis, "originais": total_antes,
                       "fracao_treinada": treinaveis / total_antes, "camadas_lora": camadas_lora},
        "antes_finetuning": antes,
        "depois_finetuning": {**depois, **metr},
        "tempo_treino_s": tempo, "memoria_gpu_gb": memoria,
        "arquivo_lora_mb": mb_lora, "modelo_float32_mb": mb_modelo,
        "diferenca_apos_mesclar": diferenca,
        "historico": historico,
        "cap6_acc_teste": (cap6 or {}).get("depois_finetuning", {}).get("acc_teste"),
    }
    nome = "lora_resultados.json" if args.rank == 16 else f"lora_resultados_r{args.rank}.json"
    arquivo = RESULTADOS_DIR / nome
    arquivo.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados salvos em: {arquivo.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
