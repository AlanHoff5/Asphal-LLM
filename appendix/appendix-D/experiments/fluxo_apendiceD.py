# Appendix D - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM

# Demonstra o Apendice D (Adding bells and whistles to the training loop) e
# separa o efeito de cada tecnica num experimento com tres treinos.

# Secoes reproduzidas do livro:
#   D.1 learning rate warmup (cronograma com 20% de aquecimento)
#   D.2 cosine decay (descida em meia onda de cosseno)
#   D.3 gradient clipping (maior gradiente e norma antes/depois do corte)
#   D.4 o treino modificado (train_model, em src/training/treino.py)

# Experimento (D.4): o mesmo GPT, o mesmo lr de pico (0.001) e 15 epocas,
# variando so o que o apendice acrescenta:
#   constante           train_model_simple, lr fixo em 0.001 (sem extras)
#   warmup_cosine       train_model sem gradient clipping
#   warmup_cosine_clip  train_model completo (o do livro)

# Uso (a partir da raiz do repositorio):
#    python Apendices/appendix-D/experiments/fluxo_apendiceD.py                 # GPT-2 124M, como no livro
#    python Apendices/appendix-D/experiments/fluxo_apendiceD.py --config mini   # versao rapida
#    python Apendices/appendix-D/experiments/fluxo_apendiceD.py --so warmup_cosine_clip

# Saidas em experiments/resultados/:
#    apendiceD_resultados.json      cronogramas, gradientes e perdas de cada treino
#    taxa_aprendizado.png           D.1 + D.2: warmup sozinho x warmup + cosine
#    comparacao_validacao.png       perda de validacao dos tres treinos
#    comparacao_treino.png          perda de treino dos tres treinos


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
from src.model.gpt_model import GPTModel
from src.training.dados import criar_loaders_treino_validacao
from src.training.dispositivo import descrever_dispositivo, escolher_dispositivo
from src.training.gradientes import find_highest_gradient, norma_do_gradiente
from src.training.graficos import plot_comparacao
from src.training.losses import calc_loss_batch, calc_loss_loader
from src.training.treino import taxa_de_aprendizado, train_model, train_model_simple

RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS = {}

CONFIGS = {
    # O modelo do livro: GPT-2 124M com contexto de 256.
    "124m": criar_config(GPT_CONFIG_124M, context_length=256),
    # Versao reduzida do projeto (roda em segundos ate na CPU).
    "mini": criar_config(GPT_CONFIG_MINI),
}

VARIANTES = {
    "constante": "lr fixo (train_model_simple)",
    "warmup_cosine": "warmup + cosine, sem clipping",
    "warmup_cosine_clip": "warmup + cosine + clipping 1.0 (livro)",
}


def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def tabela(linhas, colunas):
    larguras = [max(len(str(c)), *(len(str(l[i])) for l in linhas)) for i, c in enumerate(colunas)]
    print("  ".join(str(c).ljust(w) for c, w in zip(colunas, larguras)))
    print("  ".join("-" * w for w in larguras))
    for l in linhas:
        print("  ".join(str(v).ljust(w) for v, w in zip(l, larguras)))


# --------------------------------
# D.1 + D.2 Cronograma do lr     |
# --------------------------------

def secao_cronograma(passos_por_epoca, n_epochs=15, initial_lr=0.0001, peak_lr=0.01, warmup=0.2):
    titulo("D.1 + D.2 Warmup e cosine decay (valores do livro)")

    total_steps = passos_por_epoca * n_epochs
    warmup_steps = int(warmup * total_steps)
    min_lr = 0.1 * initial_lr
    incremento = (peak_lr - initial_lr) / warmup_steps

    # D.1: warmup e depois lr constante no pico
    so_warmup = [initial_lr + s * incremento if s < warmup_steps else peak_lr
                 for s in range(total_steps)]
    # D.2: warmup e depois cosine decay (a funcao usada no treino do projeto)
    com_cosine = [taxa_de_aprendizado(s, peak_lr, initial_lr, min_lr, warmup_steps, total_steps)
                  for s in range(total_steps)]

    print(f"{passos_por_epoca} lotes por epoca x {n_epochs} epocas = {total_steps} passos")
    print(f"warmup_steps = int(0.2 * {total_steps}) = {warmup_steps}   (livro: 27)")
    meio = warmup_steps + (total_steps - warmup_steps) // 2
    tabela([
        (0, f"{so_warmup[0]:.2e}", f"{com_cosine[0]:.2e}", "inicio do warmup"),
        (warmup_steps // 2, f"{so_warmup[warmup_steps // 2]:.2e}", f"{com_cosine[warmup_steps // 2]:.2e}", "meio do warmup"),
        (warmup_steps, f"{so_warmup[warmup_steps]:.2e}", f"{com_cosine[warmup_steps]:.2e}", "pico"),
        (meio, f"{so_warmup[meio]:.2e}", f"{com_cosine[meio]:.2e}", "meio do cosine"),
        (total_steps - 1, f"{so_warmup[-1]:.2e}", f"{com_cosine[-1]:.2e}", "ultimo passo"),
    ], ["passo", "so warmup (D.1)", "warmup + cosine (D.2)", "momento"])

    RESULTADOS["cronograma"] = {
        "total_steps": total_steps, "warmup_steps": warmup_steps,
        "initial_lr": initial_lr, "peak_lr": peak_lr, "min_lr": min_lr,
        "so_warmup": so_warmup, "warmup_cosine": com_cosine,
    }
    return so_warmup, com_cosine


def salvar_grafico_cronograma(so_warmup, com_cosine):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nao instalado: grafico do cronograma nao gerado")
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(so_warmup, label="D.1: warmup e depois constante")
    ax.plot(com_cosine, label="D.2: warmup e depois cosine decay")
    ax.set_xlabel("Passo")
    ax.set_ylabel("Taxa de aprendizado")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    arquivo = RESULTADOS_DIR / "taxa_aprendizado.png"
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)
    print(f"Grafico: {arquivo.relative_to(PROJECT_ROOT)}")


# ------------------------------
# D.3 Gradient clipping        |
# ------------------------------

def secao_clipping(cfg, train_loader, device, seed):
    titulo("D.3 Gradient clipping (um lote, modelo sem treino)")

    torch.manual_seed(seed)
    model = GPTModel(cfg).to(device)
    input_batch, target_batch = next(iter(train_loader))
    loss = calc_loss_batch(input_batch, target_batch, model, device)
    loss.backward()

    maior_antes = find_highest_gradient(model).item()
    norma_antes = norma_do_gradiente(model)
    norma_devolvida = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    maior_depois = find_highest_gradient(model).item()
    norma_depois = norma_do_gradiente(model)

    tabela([
        ("maior gradiente individual", f"{maior_antes:.4f}", f"{maior_depois:.4f}"),
        ("norma L2 de todos os gradientes", f"{norma_antes:.4f}", f"{norma_depois:.4f}"),
    ], ["medida", "antes", "depois de clip_grad_norm_(1.0)"])
    fator = 1.0 / norma_antes if norma_antes > 1.0 else 1.0
    print(f"clip_grad_norm_ devolve a norma ORIGINAL ({norma_devolvida:.4f}).")
    if norma_antes > 1.0:
        print(f"Como ela passa de 1.0, todos os gradientes sao multiplicados por 1.0 / {norma_antes:.4f} = {fator:.4f}.")
    else:
        print("Como ela ja e menor que 1.0, o corte nao muda nada neste lote.")
    print("Livro (GPU Apple, GPT-2 124M): maior gradiente 0.0446 -> 0.0201.")

    RESULTADOS["clipping"] = {
        "maior_gradiente_antes": maior_antes, "maior_gradiente_depois": maior_depois,
        "norma_antes": norma_antes, "norma_depois": norma_depois, "fator": fator,
    }


# ------------------------------
# D.4 Treino com e sem extras  |
# ------------------------------

def treinar_variante(nome, cfg, args, train_loader, val_loader, tokenizer, device):
    titulo(f"D.4 Treino: {nome} ({VARIANTES[nome]})")

    torch.manual_seed(args.seed)
    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    comum = dict(num_epochs=args.epocas, eval_freq=args.eval_freq, eval_iter=args.eval_iter,
                 start_context="Every effort moves you", tokenizer=tokenizer)
    warmup_steps = int(args.warmup * args.epocas * len(train_loader))

    inicio = time.time()
    if nome == "constante":
        historico = train_model_simple(model, train_loader, val_loader, optimizer, device, **comum)
        historico["lrs"] = [args.lr] * (args.epocas * len(train_loader))
    else:
        grad_clip = 1.0 if nome == "warmup_cosine_clip" else None
        historico = train_model(model, train_loader, val_loader, optimizer, device,
                                warmup_steps=warmup_steps, initial_lr=args.initial_lr,
                                min_lr=args.min_lr, grad_clip=grad_clip, **comum)
    tempo = time.time() - inicio

    model.eval()
    treino = calc_loss_loader(train_loader, model, device)
    val = calc_loss_loader(val_loader, model, device)
    i_min = min(range(len(historico["val_losses"])), key=historico["val_losses"].__getitem__)
    return {
        "descricao": VARIANTES[nome], "warmup_steps": warmup_steps if nome != "constante" else 0,
        "treino_final": treino, "validacao_final": val,
        "melhor_validacao": historico["val_losses"][i_min],
        "melhor_validacao_epoca": historico["epocas"][i_min],
        "maior_perda_treino_registrada": max(historico["train_losses"]),
        "tempo_s": tempo, "amostra_final": historico["amostras"][-1]["texto"],
        "historico": historico,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=sorted(CONFIGS), default="124m")
    ap.add_argument("--dados", default=str(PROJECT_ROOT / "data" / "the-verdict.txt"))
    ap.add_argument("--epocas", type=int, default=15, help="livro: 15")
    ap.add_argument("--lr", type=float, default=0.001, help="lr de pico (livro: 0.001)")
    ap.add_argument("--initial-lr", type=float, default=1e-5, help="lr no inicio do warmup (livro: 1e-5)")
    ap.add_argument("--min-lr", type=float, default=1e-5, help="lr no fim do cosine (livro: 1e-5)")
    ap.add_argument("--warmup", type=float, default=0.2, help="fracao de passos de warmup (livro: 0.2)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--eval-freq", type=int, default=5)
    ap.add_argument("--eval-iter", type=int, default=1, help="livro: 1")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="auto", help="auto, cuda ou cpu")
    ap.add_argument("--so", nargs="+", choices=list(VARIANTES), default=list(VARIANTES),
                    help="roda so os treinos citados")
    args = ap.parse_args()

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = CONFIGS[args.config]
    sufixo = "" if args.config == "124m" else f"_{args.config}"   # nao sobrescreve o 124m
    device = escolher_dispositivo(args.device)
    print(f"Dispositivo: {descrever_dispositivo(device)} | modelo: {args.config} | PyTorch {torch.__version__}")

    tokenizer = get_bpe_tokenizer()
    texto = Path(args.dados).read_text(encoding="utf-8")
    torch.manual_seed(args.seed)
    train_loader, val_loader = criar_loaders_treino_validacao(texto, tokenizer, cfg,
                                                              batch_size=args.batch_size)

    so_warmup, com_cosine = secao_cronograma(len(train_loader))
    salvar_grafico_cronograma(so_warmup, com_cosine)
    secao_clipping(cfg, train_loader, device, args.seed)

    # Com --so, os treinos que ja estao no JSON (de execucoes anteriores com a
    # mesma config) sao mantidos: da para rodar um treino por vez.
    arquivo = RESULTADOS_DIR / f"apendiceD_resultados{sufixo}.json"
    treinos = {}
    if arquivo.exists() and set(args.so) != set(VARIANTES):
        anteriores = json.loads(arquivo.read_text(encoding="utf-8")).get("treinos", {})
        treinos = {n: r for n, r in anteriores.items() if n not in args.so}
        if treinos:
            print(f"Mantendo do JSON anterior: {', '.join(treinos)}")
    for nome in args.so:
        treinos[nome] = treinar_variante(nome, cfg, args, train_loader, val_loader, tokenizer, device)
    treinos = {n: treinos[n] for n in VARIANTES if n in treinos}      # ordem fixa

    titulo("Resumo dos treinos (mesmo lr de pico, mesma semente)")
    tabela([(n, f"{r['treino_final']:.3f}", f"{r['validacao_final']:.3f}",
             f"{r['melhor_validacao']:.3f} (ep {r['melhor_validacao_epoca']:.1f})",
             f"{r['maior_perda_treino_registrada']:.2f}", f"{r['tempo_s']:.0f} s")
            for n, r in treinos.items()],
           ["treino", "perda treino", "perda val.", "melhor val.", "maior perda registrada", "tempo"])
    for n, r in treinos.items():
        print(f"\n[{n}] {r['amostra_final']!r}")

    if len(treinos) > 1:
        series = {n: r["historico"] for n, r in treinos.items()}
        for chave, nome_arq, rotulo in (("val_losses", f"comparacao_validacao{sufixo}.png", "validacao"),
                                        ("train_losses", f"comparacao_treino{sufixo}.png", "treino")):
            try:
                plot_comparacao(series, chave, RESULTADOS_DIR / nome_arq,
                                f"Apendice D: perda de {rotulo} ({args.config}, lr de pico {args.lr})")
            except ImportError:
                print("matplotlib nao instalado: graficos de comparacao nao gerados")
                break

    RESULTADOS["parametros"] = {
        "config": args.config, "modelo": cfg, "epocas": args.epocas, "lr_pico": args.lr,
        "initial_lr": args.initial_lr, "min_lr": args.min_lr, "warmup": args.warmup,
        "batch_size": args.batch_size, "eval_freq": args.eval_freq, "eval_iter": args.eval_iter,
        "seed": args.seed, "lotes_por_epoca": len(train_loader),
        "dispositivo": descrever_dispositivo(device), "torch": str(torch.__version__),
    }
    RESULTADOS["treinos"] = treinos
    arquivo.write_text(json.dumps(RESULTADOS, indent=2, ensure_ascii=False), encoding="utf-8")
    titulo("FLUXO DO APENDICE D CONCLUIDO")
    print(f"Resultados salvos em {arquivo.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
