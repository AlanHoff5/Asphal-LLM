"""Sprint 5 - Entregavel 4: experimentos de treinamento.

Roda varios treinos em sequencia, mudando UM parametro de cada vez em
relacao ao treino do livro (baseline), e monta a comparacao:

    exp01_baseline            treino do livro (lr 0.0004, 10 epocas, dropout 0.1, wd 0.1)
    exp02_lr_alta             taxa de aprendizado 0.001
    exp03_lr_baixa            taxa de aprendizado 0.0001
    exp04_30_epocas           30 epocas (mais overfitting?)
    exp05_sem_regularizacao   sem dropout e sem weight decay
    exp06_warmup_cosine_clip  lr 0.001 com warmup, cosine decay e gradient clipping (apendice D)
    exp07_modelo_mini         modelo pequeno do projeto (GPT_CONFIG_MINI)
    exp08_modelo_small        modelo intermediario (~30M parametros)

Cada experimento roda o treino_modelo.py em um processo separado (assim a
memoria da GPU e liberada entre um e outro). Os pesos NAO sao salvos aqui
(cada arquivo do 124M tem ~2 GB); o modelo salvo para os Entregaveis 5 e 6
e o models/ch05/treino_124m.pth.

Uso (a partir da raiz do repositorio):
    python sprints/ch05/experiments/experimentos_treino.py            # roda o que falta
    python sprints/ch05/experiments/experimentos_treino.py --refazer  # roda tudo de novo
    python sprints/ch05/experiments/experimentos_treino.py --so exp02_lr_alta exp03_lr_baixa
    python sprints/ch05/experiments/experimentos_treino.py --so-resumo  # so refaz a tabela

Saidas em experiments/resultados/:
    expNN_*.json / .png              resultado de cada experimento
    experimentos_resumo.md           tabela comparativa + amostras de texto
    comparacao_validacao.png         perda de validacao de todos os experimentos
    comparacao_treino.png            perda de treino de todos os experimentos
"""

from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.graficos import plot_comparacao

RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
SCRIPT_TREINO = Path(__file__).resolve().parent / "treino_modelo.py"

# (nome, o que muda em relacao ao baseline, argumentos extras do treino_modelo.py)
EXPERIMENTOS = [
    ("exp01_baseline", "treino do livro", []),
    ("exp02_lr_alta", "lr 0.0004 -> 0.001", ["--lr", "0.001"]),
    ("exp03_lr_baixa", "lr 0.0004 -> 0.0001", ["--lr", "0.0001"]),
    ("exp04_30_epocas", "10 -> 30 epocas", ["--epocas", "30"]),
    ("exp05_sem_regularizacao", "dropout 0 e weight decay 0", ["--drop-rate", "0", "--weight-decay", "0"]),
    ("exp06_warmup_cosine_clip", "lr 0.001 + warmup 20% + cosine + clip 1.0",
     ["--lr", "0.001", "--warmup", "0.2", "--grad-clip", "1.0"]),
    ("exp07_modelo_mini", "modelo mini (emb 128, 4 camadas, contexto 128)", ["--config", "mini"]),
    ("exp08_modelo_small", "modelo small (emb 256, 6 camadas, contexto 256)", ["--config", "small"]),
]


def rodar(nome, extras):
    comando = [sys.executable, str(SCRIPT_TREINO), "--nome", nome, "--nao-salvar-modelo", *extras]
    print("\n" + "#" * 78)
    print(f"# {nome}: {' '.join(extras) or '(baseline)'}")
    print("#" * 78, flush=True)
    inicio = time.time()
    codigo = subprocess.call(comando, cwd=PROJECT_ROOT)
    print(f"# {nome} terminou em {(time.time() - inicio) / 60:.1f} min (codigo {codigo})", flush=True)
    return codigo == 0


def avaliar_overfitting(r):
    """Leitura simples do overfitting a partir das perdas."""
    p = r["perdas"]
    gap = p["validacao_final"] - p["treino_final"]
    subiu = p["validacao_final"] - p["melhor_validacao"]["val_loss"]
    if subiu > 0.1 and gap > 1.0:
        return f"sim: val subiu {subiu:.2f} depois do minimo (gap {gap:.2f})"
    if gap > 1.0:
        return f"gap grande ({gap:.2f})"
    return f"nao evidente (gap {gap:.2f})"


def montar_resumo():
    linhas = [
        "# Resumo dos experimentos de treinamento (Sprint 5)",
        "",
        "Todos usam The Verdict (90% treino / 10% validacao), AdamW e seed 123. "
        "Cada experimento muda um parametro em relacao ao baseline.",
        "",
        "| experimento | o que mudou | parametros | perda treino | perda val | menor val (epoca) "
        "| gap val-treino | ppl val | tempo | tokens/s | GPU (GB) | overfitting? |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    series, amostras = {}, []
    for nome, mudanca, _ in EXPERIMENTOS:
        arquivo = RESULTADOS_DIR / f"{nome}.json"
        if not arquivo.exists():
            linhas.append(f"| {nome} | {mudanca} | (nao rodou) |" + " |" * 9)
            continue
        r = json.loads(arquivo.read_text(encoding="utf-8"))
        p = r["perdas"]
        m = p["melhor_validacao"]
        memoria = f"{r['memoria_gpu_gb']:.2f}" if r.get("memoria_gpu_gb") else "-"
        linhas.append(
            f"| {nome} | {mudanca} | {r['parametros']['num_parametros'] / 1e6:.1f}M "
            f"| {p['treino_final']:.3f} | {p['validacao_final']:.3f} | {m['val_loss']:.3f} ({m['epoca']:.1f}) "
            f"| {p['validacao_final'] - p['treino_final']:.3f} | {p['perplexity_validacao_final']:,.1f} "
            f"| {r['tempo_total_s'] / 60:.1f} min | {r['tokens_por_segundo']:,.0f} | {memoria} "
            f"| {avaliar_overfitting(r)} |"
        )
        series[nome] = r["historico"]
        ultima = r["amostras"][-1]["texto"].replace("\n", " ")
        amostras.append(f"- **{nome}** (epoca {r['amostras'][-1]['epoca']}): {ultima}")

    linhas += ["", "## Ultima amostra de texto de cada experimento",
               "", "Contexto inicial: \"Every effort moves you\" (decodificacao gulosa).", ""] + amostras
    (RESULTADOS_DIR / "experimentos_resumo.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas[:len(EXPERIMENTOS) + 6]))

    if series:
        plot_comparacao(series, "val_losses", RESULTADOS_DIR / "comparacao_validacao.png",
                        "Perda de validacao por experimento")
        plot_comparacao(series, "train_losses", RESULTADOS_DIR / "comparacao_treino.png",
                        "Perda de treino por experimento")
    print(f"\nResumo: {(RESULTADOS_DIR / 'experimentos_resumo.md').relative_to(PROJECT_ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refazer", action="store_true", help="roda de novo mesmo se ja existir resultado")
    ap.add_argument("--so", nargs="+", metavar="NOME", help="roda so estes experimentos")
    ap.add_argument("--so-resumo", action="store_true", help="nao treina, so monta a tabela e os graficos")
    args = ap.parse_args()

    if not args.so_resumo:
        inicio = time.time()
        for nome, _, extras in EXPERIMENTOS:
            if args.so and nome not in args.so:
                continue
            if (RESULTADOS_DIR / f"{nome}.json").exists() and not args.refazer:
                print(f"Pulando {nome}: resultado ja existe (use --refazer para rodar de novo)")
                continue
            if not rodar(nome, extras):
                print(f"AVISO: {nome} falhou; seguindo para o proximo.")
        print(f"\nTempo total dos experimentos: {(time.time() - inicio) / 60:.1f} min")

    montar_resumo()


if __name__ == "__main__":
    main()
