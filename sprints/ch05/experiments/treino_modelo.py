# Sprint 5 - Entregavel 3: treinamento do GPT (secao 5.2).

# Treina o modelo com o laco do livro (train_model_simple) e registra:
#  - os parametros usados (modelo, epocas, lote, taxa de aprendizado, otimizador)
#  - as perdas de treino e validacao ao longo do treino
#  - um texto de exemplo ao fim de cada epoca
#  - tempo total, tokens por segundo e memoria de GPU usada
#  - o grafico das perdas e os pesos finais do modelo

# Uso (a partir da raiz do repositorio):
#    python sprints/ch05/experiments/treino_modelo.py                 # igual ao livro
#    python sprints/ch05/experiments/treino_modelo.py --config mini   # versao rapida
#    python sprints/ch05/experiments/treino_modelo.py --epocas 20 --lr 0.001 --nome lr_alta
#    python sprints/ch05/experiments/treino_modelo.py --lr 0.001 --warmup 0.2 --grad-clip 1.0

# Saidas:
#    experiments/resultados/<nome>.json   parametros, perdas, amostras e tempos
#    experiments/resultados/<nome>.png    grafico treino x validacao
#    models/ch05/<nome>.pth               pesos + otimizador (fora do git)


from pathlib import Path
import argparse
import json
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import get_bpe_tokenizer
from src.model.config import GPT_CONFIG_124M, GPT_CONFIG_MINI, criar_config
from src.model.gpt_model import GPTModel, contar_parametros
from src.training.dados import contar_tokens_do_loader, criar_loaders_treino_validacao
from src.training.dispositivo import descrever_dispositivo, escolher_dispositivo
from src.training.graficos import plot_losses
from src.training.losses import calc_loss_loader, perplexidade
from src.training.treino import train_model, train_model_simple

RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
MODELOS_DIR = PROJECT_ROOT / "models" / "ch05"

CONFIGS = {
    # Exatamente o modelo do capitulo 5: GPT-2 124M com contexto de 256.
    "124m": criar_config(GPT_CONFIG_124M, context_length=256),
    # Versao reduzida do projeto (roda em segundos ate na CPU).
    "mini": criar_config(GPT_CONFIG_MINI),
    # Meio-termo: ~30M parametros.
    "small": criar_config(GPT_CONFIG_MINI, context_length=256, emb_dim=256, n_heads=8, n_layers=6),
}


def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def medir_perdas_completas(model, train_loader, val_loader, device):
    # Perda em TODOS os lotes (nao so eval_iter), para o antes/depois.
    model.eval()
    treino = calc_loss_loader(train_loader, model, device)
    val = calc_loss_loader(val_loader, model, device)
    return treino, val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=sorted(CONFIGS), default="124m")
    ap.add_argument("--dados", default=str(PROJECT_ROOT / "data" / "the-verdict.txt"))
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.0004)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--drop-rate", type=float, default=None, help="padrao: o da config (0.1)")
    ap.add_argument("--eval-freq", type=int, default=5)
    ap.add_argument("--eval-iter", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--contexto-inicial", default="Every effort moves you")
    ap.add_argument("--device", default="auto", help="auto, cuda ou cpu")
    ap.add_argument("--nome", default=None, help="nome dos arquivos de saida")
    ap.add_argument("--nao-salvar-modelo", action="store_true")
    # Apendice D: com --warmup o treino usa train_model (warmup + cosine + clipping)
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="fracao dos passos de aquecimento do lr (ex.: 0.2); 0 = treino do cap. 5")
    ap.add_argument("--initial-lr", type=float, default=0.0001, help="lr no inicio do warmup")
    ap.add_argument("--min-lr", type=float, default=0.00001, help="lr no fim do cosine decay")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="norma maxima do gradiente (com --warmup)")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    if args.drop_rate is not None:
        cfg = criar_config(cfg, drop_rate=args.drop_rate)
    nome = args.nome or f"treino_{args.config}"

    # --------------------------------------------------------
    # 1. Preparacao: dispositivo, dados, modelo e otimizador |
    # --------------------------------------------------------
    titulo("1. Preparacao")
    device = escolher_dispositivo(args.device)
    print(f"Dispositivo: {descrever_dispositivo(device)}")

    tokenizer = get_bpe_tokenizer()
    texto = Path(args.dados).read_text(encoding="utf-8")
    torch.manual_seed(args.seed)
    train_loader, val_loader = criar_loaders_treino_validacao(
        texto, tokenizer, cfg, batch_size=args.batch_size, train_ratio=0.90
    )

    torch.manual_seed(args.seed)
    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    parametros = {
        "config": args.config, "modelo": cfg, "num_parametros": contar_parametros(model),
        "dados": args.dados, "tokens_treino": contar_tokens_do_loader(train_loader),
        "tokens_validacao": contar_tokens_do_loader(val_loader),
        "lotes_por_epoca": len(train_loader), "epocas": args.epocas,
        "batch_size": args.batch_size, "otimizador": "AdamW", "lr": args.lr,
        "weight_decay": args.weight_decay, "eval_freq": args.eval_freq,
        "eval_iter": args.eval_iter, "seed": args.seed,
        "drop_rate": cfg["drop_rate"],
        "agendamento_lr": (f"warmup {args.warmup:.0%} + cosine ({args.initial_lr} -> {args.lr} -> {args.min_lr}), "
                           f"grad_clip {args.grad_clip}") if args.warmup > 0 else "constante",
        "dispositivo": descrever_dispositivo(device), "torch": str(torch.__version__),
    }
    for chave, valor in parametros.items():
        print(f"  {chave:<18} {valor}")

    # --------------------------------------
    # 2. Estado inicial (pesos aleatorios) |
    # --------------------------------------
    titulo("2. Antes do treino")
    perda_treino_ini, perda_val_ini = medir_perdas_completas(model, train_loader, val_loader, device)
    print(f"Perda treino {perda_treino_ini:.3f} | validacao {perda_val_ini:.3f} | "
          f"perplexity val {perplexidade(perda_val_ini):,.0f}")

    # -----------
    # 3. Treino |
    # -----------
    titulo(f"3. Treinando por {args.epocas} epocas ({len(train_loader)} lotes por epoca)")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inicio = time.time()
    try:
        comum = dict(num_epochs=args.epocas, eval_freq=args.eval_freq, eval_iter=args.eval_iter,
                     start_context=args.contexto_inicial, tokenizer=tokenizer)
        if args.warmup > 0:
            warmup_steps = int(args.warmup * args.epocas * len(train_loader))
            historico = train_model(model, train_loader, val_loader, optimizer, device,
                                    warmup_steps=warmup_steps, initial_lr=args.initial_lr,
                                    min_lr=args.min_lr, grad_clip=args.grad_clip, **comum)
        else:
            historico = train_model_simple(model, train_loader, val_loader, optimizer, device, **comum)
    except torch.cuda.OutOfMemoryError:
        print("\nERRO: a memoria da GPU acabou. Rode de novo com um --batch-size menor "
              "ou com --config small / mini.")
        sys.exit(1)
    tempo_total = time.time() - inicio

    # --------------
    # 4. Resultado |
    # --------------
    titulo("4. Depois do treino")
    perda_treino_fim, perda_val_fim = medir_perdas_completas(model, train_loader, val_loader, device)
    tokens_treinados = parametros["tokens_treino"] * args.epocas
    print(f"{'':<12}{'antes':>10}{'depois':>10}{'ppl depois':>14}")
    print(f"{'treino':<12}{perda_treino_ini:>10.3f}{perda_treino_fim:>10.3f}{perplexidade(perda_treino_fim):>14,.1f}")
    print(f"{'validacao':<12}{perda_val_ini:>10.3f}{perda_val_fim:>10.3f}{perplexidade(perda_val_fim):>14,.1f}")
    print(f"\nTempo total: {tempo_total / 60:.1f} min | {tokens_treinados / tempo_total:,.0f} tokens/s")
    memoria_gpu = None
    if device.type == "cuda":
        memoria_gpu = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        print(f"Memoria maxima usada na GPU: {memoria_gpu:.2f} GB")
    print(f"Diferenca validacao - treino: {perda_val_fim - perda_treino_fim:.3f} "
          "(quanto maior, mais o modelo decorou o texto de treino)")
    i_melhor = min(range(len(historico["val_losses"])), key=historico["val_losses"].__getitem__)
    melhor = {"val_loss": historico["val_losses"][i_melhor], "step": historico["steps"][i_melhor],
              "epoca": historico["epocas"][i_melhor]}
    if perda_val_fim <= melhor["val_loss"]:
        # a medida final (todos os lotes) e a menor: a validacao ainda estava caindo
        melhor = {"val_loss": perda_val_fim, "step": historico["steps"][-1], "epoca": float(args.epocas)}
        print(f"Menor perda de validacao: {melhor['val_loss']:.3f} no fim do treino "
              "(a validacao ainda estava caindo: mais epocas poderiam ajudar)")
    else:
        print(f"Menor perda de validacao: {melhor['val_loss']:.3f} no passo {melhor['step']} "
              f"(epoca {melhor['epoca']:.1f}); depois disso a validacao piorou (overfitting)")

    # ---------------------------------------
    # 5. Salvar resultados, grafico e pesos |
    # ---------------------------------------
    titulo("5. Arquivos gerados")
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    grafico = plot_losses(historico["epocas"], historico["tokens_seen"],
                          historico["train_losses"], historico["val_losses"],
                          RESULTADOS_DIR / f"{nome}.png", titulo=nome)
    print(f"Grafico:    {Path(grafico).relative_to(PROJECT_ROOT)}")

    resultado = {
        "parametros": parametros,
        "perdas": {"treino_inicial": perda_treino_ini, "validacao_inicial": perda_val_ini,
                   "treino_final": perda_treino_fim, "validacao_final": perda_val_fim,
                   "perplexity_validacao_final": perplexidade(perda_val_fim),
                   "melhor_validacao": melhor},
        "tempo_total_s": tempo_total, "tempo_por_epoca_s": historico["tempo_por_epoca_s"],
        "tokens_por_segundo": tokens_treinados / tempo_total, "memoria_gpu_gb": memoria_gpu,
        "historico": {k: historico[k] for k in ("steps", "epocas", "tokens_seen", "train_losses", "val_losses", "lrs")
                      if k in historico},
        "amostras": historico["amostras"],
    }
    arquivo_json = RESULTADOS_DIR / f"{nome}.json"
    arquivo_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados: {arquivo_json.relative_to(PROJECT_ROOT)}")

    if not args.nao_salvar_modelo:
        MODELOS_DIR.mkdir(parents=True, exist_ok=True)
        arquivo_pesos = MODELOS_DIR / f"{nome}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": cfg,
            "parametros": parametros,
        }, arquivo_pesos)
        print(f"Pesos:      {arquivo_pesos.relative_to(PROJECT_ROOT)} (fora do git)")


if __name__ == "__main__":
    main()
