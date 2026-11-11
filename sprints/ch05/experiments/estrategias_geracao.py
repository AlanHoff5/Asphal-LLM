"""Sprint 5 - Entregavel 5: estrategias de geracao de texto (secao 5.3).

Compara a escolha do proximo token por:
    gulosa (greedy)        sempre o token mais provavel (argmax)
    amostragem             sorteio pelas probabilidades (torch.multinomial)
    temperatura            divide os logits antes do softmax (T<1 concentra, T>1 espalha)
    top-k                  so os k tokens mais provaveis podem ser sorteados

Parte 1  o exemplo do livro com 9 palavras: quantas vezes cada palavra e
         sorteada em 1000 tentativas, com temperatura 1, 0.1 e 5, e o top-k=3
Parte 2  as mesmas estrategias no GPT, com o MESMO contexto inicial, em dois
         modelos: o treinado (models/ch05/treino_124m.pth) e o mesmo modelo
         SEM treino. Assim da para separar o efeito da estrategia de geracao
         do efeito do treinamento dos pesos.

Metricas de cada texto gerado (so os tokens novos):
    distinct-1 / distinct-2   fracao de palavras / pares de palavras unicos (diversidade)
    repeticao                 fracao de trigramas repetidos
    copia do treino           fracao de sequencias de 8 tokens que existem no texto de treino
                              (mede o quanto o modelo esta "decorando")
    amostras distintas        quantos textos diferentes saem de 3 sementes

Uso (a partir da raiz do repositorio, depois de rodar treino_modelo.py):
    python sprints/ch05/experiments/estrategias_geracao.py
"""

from pathlib import Path
import argparse
import json
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import get_bpe_tokenizer
from src.model.generation import aplicar_top_k, generate, text_to_token_ids, token_ids_to_text
from src.model.gpt_model import GPTModel
from src.training.checkpoint import carregar_modelo
from src.training.dados import dividir_treino_validacao
from src.training.dispositivo import descrever_dispositivo, escolher_dispositivo

RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS = {}

# (nome, temperatura, top_k). temperatura 0 = gulosa.
ESTRATEGIAS = [
    ("gulosa", 0.0, None),
    ("amostragem T=1.0", 1.0, None),
    ("temperatura T=0.5", 0.5, None),
    ("temperatura T=1.5", 1.5, None),
    ("top-k=3, T=1.0", 1.0, 3),
    ("top-k=40, T=1.0", 1.0, 40),
    ("top-k=25, T=1.4 (livro)", 1.4, 25),
]
SEMENTES = [123, 124, 125]


def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Parte 1 - Exemplo do livro com um vocabulario de 9 palavras
# ---------------------------------------------------------------------------
def parte1_exemplo_do_livro():
    titulo("Parte 1 - Exemplo do livro: 9 palavras, 1000 sorteios")
    vocab = ["closer", "every", "effort", "forward", "inches", "moves", "pizza", "toward", "you"]
    next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

    print(f"{'palavra':<9}{'logit':>7}" + "".join(f"{f'T={t}':>10}" for t in (1.0, 0.1, 5.0)) + f"{'top-k=3':>10}")
    def sortear_1000(probas):
        # igual ao print_sampled_tokens do livro: 1000 sorteios de 1 token
        torch.manual_seed(123)
        sorteios = torch.tensor([torch.multinomial(probas, num_samples=1).item() for _ in range(1000)])
        return torch.bincount(sorteios, minlength=len(vocab)).tolist()

    contagens = {}
    for t in (1.0, 0.1, 5.0):
        contagens[f"T={t}"] = sortear_1000(torch.softmax(next_token_logits / t, dim=0))
    probas_topk = torch.softmax(aplicar_top_k(next_token_logits.unsqueeze(0), 3).squeeze(0), dim=0)
    contagens["top-k=3"] = sortear_1000(probas_topk)

    for i, palavra in enumerate(vocab):
        print(f"{palavra:<9}{next_token_logits[i].item():>7.2f}"
              + "".join(f"{contagens[k][i]:>10}" for k in contagens))
    print("\nGulosa: escolheria sempre 'forward' (maior logit).")
    print("T=0.1 fica quase igual a gulosa; T=5 espalha os sorteios e ate 'pizza' aparece;")
    print("top-k=3 so deixa sortear closer, forward e toward (o resto vira probabilidade 0).")
    RESULTADOS["exemplo_livro"] = {"vocab": vocab, "contagens_em_1000": contagens,
                                   "probabilidades_top_k_3": [round(p, 4) for p in probas_topk.tolist()]}


# ---------------------------------------------------------------------------
# Parte 2 - Estrategias no GPT treinado e sem treino
# ---------------------------------------------------------------------------
def ngramas(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def metricas(novos, ngramas_treino):
    def distinct(n):
        g = ngramas(novos, n)
        return len(set(g)) / len(g) if g else 0.0
    trigramas = ngramas(novos, 3)
    oito = ngramas(novos, 8)
    return {
        "distinct_1": distinct(1),
        "distinct_2": distinct(2),
        "repeticao": 1 - len(set(trigramas)) / len(trigramas) if trigramas else 0.0,
        "copia_do_treino": sum(g in ngramas_treino for g in oito) / len(oito) if oito else 0.0,
    }


def comparar(model, nome_modelo, tokenizer, device, contexto, max_new_tokens, ngramas_treino):
    print(f"\n--- {nome_modelo} ---")
    ids = text_to_token_ids(contexto, tokenizer).to(device)
    n_contexto = ids.shape[1]
    linhas = []
    for nome, temperatura, top_k in ESTRATEGIAS:
        textos, medidas = [], []
        for semente in SEMENTES:
            torch.manual_seed(semente)
            saida = generate(model, ids, max_new_tokens, model.cfg["context_length"],
                             temperature=temperatura, top_k=top_k)
            textos.append(token_ids_to_text(saida.cpu(), tokenizer))
            medidas.append(metricas(saida[0, n_contexto:].tolist(), ngramas_treino))
        media = {k: sum(m[k] for m in medidas) / len(medidas) for k in medidas[0]}
        distintas = len(set(textos))
        print(f"\n[{nome}]  distinct-1 {media['distinct_1']:.2f} | distinct-2 {media['distinct_2']:.2f} | "
              f"repeticao {media['repeticao']:.2f} | copia do treino {media['copia_do_treino']:.2f} | "
              f"amostras distintas {distintas}/{len(SEMENTES)}")
        for i, t in enumerate(textos):
            print(f"   ({i + 1}) {t!r}")
        linhas.append({"modelo": nome_modelo, "estrategia": nome, "temperatura": temperatura,
                       "top_k": top_k, **media, "amostras_distintas": distintas, "textos": textos})
    return linhas


def salvar_markdown(linhas, contexto):
    md = ["# Estrategias de geracao (Sprint 5 - Entregavel 5)", "",
          f"Contexto inicial: `{contexto}` | sementes {SEMENTES} | metricas sobre os tokens gerados.", "",
          "| modelo | estrategia | distinct-1 | distinct-2 | repeticao | copia do treino | amostras distintas |",
          "|---|---|---|---|---|---|---|"]
    for l in linhas:
        md.append(f"| {l['modelo']} | {l['estrategia']} | {l['distinct_1']:.2f} | {l['distinct_2']:.2f} "
                  f"| {l['repeticao']:.2f} | {l['copia_do_treino']:.2f} | {l['amostras_distintas']}/{len(SEMENTES)} |")
    md += ["", "## Textos gerados", ""]
    for l in linhas:
        md.append(f"**{l['modelo']} - {l['estrategia']}**")
        md += [f"{i + 1}. {t.replace(chr(10), ' ')}" for i, t in enumerate(l["textos"])]
        md.append("")
    arquivo = RESULTADOS_DIR / "estrategias_geracao.md"
    arquivo.write_text("\n".join(md), encoding="utf-8")
    return arquivo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=str(PROJECT_ROOT / "models" / "ch05" / "treino_124m.pth"))
    ap.add_argument("--dados", default=str(PROJECT_ROOT / "data" / "the-verdict.txt"))
    ap.add_argument("--contexto", default="Every effort moves you")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    parte1_exemplo_do_livro()

    titulo("Parte 2 - Mesmas estrategias no GPT: treinado x sem treino")
    if not Path(args.modelo).exists():
        sys.exit(f"Arquivo {args.modelo} nao existe. Rode antes: python sprints/ch05/experiments/treino_modelo.py")
    device = escolher_dispositivo(args.device)
    print(f"Dispositivo: {descrever_dispositivo(device)} | contexto: {args.contexto!r}")
    tokenizer = get_bpe_tokenizer()
    texto_treino, _ = dividir_treino_validacao(Path(args.dados).read_text(encoding="utf-8"))
    ngramas_treino = set(ngramas(tokenizer.encode(texto_treino), 8))

    linhas = []
    treinado, checkpoint = carregar_modelo(args.modelo, device)
    epocas = checkpoint.get("parametros", {}).get("epocas", "?")
    linhas += comparar(treinado, f"treinado ({epocas} epocas)", tokenizer, device, args.contexto,
                       args.max_new_tokens, ngramas_treino)
    cfg = checkpoint["cfg"]
    del treinado, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    torch.manual_seed(123)
    sem_treino = GPTModel(cfg).to(device).eval()
    linhas += comparar(sem_treino, "sem treino (pesos aleatorios)", tokenizer, device, args.contexto,
                       args.max_new_tokens, ngramas_treino)

    RESULTADOS["parte2"] = linhas
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTADOS_DIR / "estrategias_geracao.json").write_text(
        json.dumps(RESULTADOS, indent=2, ensure_ascii=False), encoding="utf-8")
    md = salvar_markdown(linhas, args.contexto)
    print(f"\nResultados salvos em {md.relative_to(PROJECT_ROOT)} (e .json)")


if __name__ == "__main__":
    main()
