# Demonstra o fluxo completo de dados da Sprint 2.

# Experimentos cobertos (conforme enunciado):
#  Exp 1 - Diferentes tamanhos de contexto (max_length)
#  Exp 2 - Diferentes tamanhos de lote (batch_size)
#  Exp 3 - Diferentes dimensões de embedding
#  Exp 4 - Diferentes sequências de entrada (tokenizer customizado vs BPE)
#  Exp 5 - Quantidade de tokens produzidos para textos diferentes
#  Exp 6 - Relação entre tamanho do contexto e quantidade de amostras
#  Exp 7 - Dimensões das estruturas produzidas pelo pipeline


from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloader_v1
from src.data.dataset import GPTDatasetV1
from src.data.embeddings import GPTInputEmbeddings
from src.data.tokenizer import create_tokenizer, load_text, tokenize

CORPUS = PROJECT_ROOT / "data" / "the-verdict.txt"

# -----------------------------------------------------------
# Fluxo completo sobre texto curto (demonstração 3.1 → 3.6) |
# -----------------------------------------------------------

def mostrar_fluxo():
    texto = "um dois tres quatro cinco seis sete oito"
    max_length = 3
    stride = 1
    batch_size = 2
    embedding_dim = 5

    print("=" * 60)
    print("FLUXO COMPLETO DA SPRINT 2")
    print("=" * 60)

    print("\nTEXTO BRUTO")
    print(texto)

    tokens = tokenize(texto)
    print("\nTOKENIZACAO")
    print("Tokens:", tokens)
    print("Quantidade de tokens:", len(tokens))

    tokenizer = create_tokenizer(texto)
    token_ids = tokenizer.encode(texto)
    print("\nVOCABULARIO E TOKEN IDS")
    print("Vocabulario:", tokenizer.str_to_int)
    print("Token IDs:", token_ids)
    print("Decodificacao:", tokenizer.decode(token_ids))

    dataset = GPTDatasetV1(texto, tokenizer, max_length=max_length, stride=stride)
    primeira_entrada, primeiro_alvo = dataset[0]
    print("\nSEQUENCIAS DE TREINAMENTO")
    print("Quantidade de amostras:", len(dataset))
    print("Primeira entrada:", primeira_entrada.tolist())
    print("Primeiro alvo:", primeiro_alvo.tolist())
    print("Alvo = entrada deslocada:", torch.equal(primeiro_alvo[:-1], primeira_entrada[1:]))

    dataloader = create_dataloader_v1(
        texto, tokenizer,
        batch_size=batch_size, max_length=max_length,
        stride=stride, shuffle=False, drop_last=False,
    )
    inputs, targets = next(iter(dataloader))
    print("\nDATALOADER")
    print("Lotes:", len(dataloader))
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)
    print("Shape dos inputs:", tuple(inputs.shape))

    torch.manual_seed(7)
    emb = GPTInputEmbeddings(
        vocab_size=len(tokenizer.str_to_int),
        context_length=max_length,
        embedding_dim=embedding_dim,
    )
    tok_emb = emb.token_embedding(inputs)
    pos_emb = emb.position_embedding(torch.arange(max_length))
    model_input = emb(inputs)

    print("\nEMBEDDINGS")
    print("Token embeddings shape:", tuple(tok_emb.shape))
    print("Positional embeddings shape:", tuple(pos_emb.shape))
    print("Entrada final shape:", tuple(model_input.shape))

    # 3.5 — a posição altera a representação mesmo com token id idêntico
    ids_rep = torch.tensor([[token_ids[0]] * max_length])
    tok_puro = emb.token_embedding(ids_rep)[0]
    saida_rep = emb(ids_rep)[0]
    print("\nPOSICAO ALTERA A REPRESENTACAO (3.5)")
    print("Token embedding puro igual nas posicoes 0 e 1:",
          torch.allclose(tok_puro[0], tok_puro[1]))
    print("Entrada final igual nas posicoes 0 e 1:",
          torch.allclose(saida_rep[0], saida_rep[1]))

    print("\nFLUXO CONCLUIDO")


# ------------------------------------
# Fluxo resumido sobre o corpus real |
# ------------------------------------

def mostrar_fluxo_corpus():
    max_length = 8
    stride = 8
    batch_size = 8
    embedding_dim = 256

    texto = load_text(CORPUS)
    tokens = tokenize(texto)
    tokenizer = create_tokenizer(texto)

    print("\n" + "=" * 60)
    print("FLUXO SOBRE O CORPUS (data/the-verdict.txt)")
    print("=" * 60)
    print("Caracteres:", len(texto))
    print("Tokens (regex):", len(tokens))
    print("Vocabulario:", len(tokenizer.str_to_int))

    dataloader = create_dataloader_v1(
        texto, tokenizer,
        batch_size=batch_size, max_length=max_length,
        stride=stride, shuffle=False, drop_last=True,
    )
    inputs, targets = next(iter(dataloader))
    print("Lotes:", len(dataloader))
    print("Shape inputs:", tuple(inputs.shape))
    print("Shape targets:", tuple(targets.shape))

    torch.manual_seed(7)
    emb = GPTInputEmbeddings(
        vocab_size=len(tokenizer.str_to_int),
        context_length=max_length,
        embedding_dim=embedding_dim,
    )
    model_input = emb(inputs)
    print("Shape entrada Transformer:", tuple(model_input.shape))


# ------------------------------------------------------
# Exp 1 — Diferentes tamanhos de contexto (max_length) |
# ------------------------------------------------------

def exp1_tamanhos_de_contexto():
    # Varia max_length e observa como muda o shape do lote.
    print("\n" + "=" * 60)
    print("EXP 1 — Diferentes tamanhos de contexto")
    print("=" * 60)

    texto = load_text(CORPUS)
    tokenizer = create_tokenizer(texto)

    configs = [4, 8, 16, 64, 256]
    batch_size = 4
    embedding_dim = 64

    print(f"{'max_length':>12}  {'amostras':>10}  {'lotes':>8}  "
          f"{'shape_input':>14}  {'shape_emb':>18}")

    for ml in configs:
        dl = create_dataloader_v1(
            texto, tokenizer,
            batch_size=batch_size, max_length=ml,
            stride=ml, shuffle=False, drop_last=True,
        )
        inputs, _ = next(iter(dl))
        torch.manual_seed(7)
        emb = GPTInputEmbeddings(
            vocab_size=len(tokenizer.str_to_int),
            context_length=ml,
            embedding_dim=embedding_dim,
        )
        out = emb(inputs)
        total_amostras = len(dl) * batch_size
        print(f"{ml:>12}  {total_amostras:>10}  {len(dl):>8}  "
              f"{str(tuple(inputs.shape)):>14}  {str(tuple(out.shape)):>18}")


# --------------------------------------------------
# Exp 2 — Diferentes tamanhos de lote (batch_size) |
# --------------------------------------------------

def exp2_tamanhos_de_lote():
    # Varia batch_size e mostra a mudança no shape do lote.
    print("\n" + "=" * 60)
    print("EXP 2 — Diferentes tamanhos de lote")
    print("=" * 60)

    texto = load_text(CORPUS)
    tokenizer = create_tokenizer(texto)
    ml = 8
    embedding_dim = 64

    print(f"{'batch_size':>12}  {'lotes':>8}  {'shape_input':>14}  {'shape_emb':>22}")

    for bs in [1, 4, 8, 32, 64]:
        dl = create_dataloader_v1(
            texto, tokenizer,
            batch_size=bs, max_length=ml,
            stride=ml, shuffle=False, drop_last=True,
        )
        inputs, _ = next(iter(dl))
        torch.manual_seed(7)
        emb = GPTInputEmbeddings(
            vocab_size=len(tokenizer.str_to_int),
            context_length=ml,
            embedding_dim=embedding_dim,
        )
        out = emb(inputs)
        print(f"{bs:>12}  {len(dl):>8}  {str(tuple(inputs.shape)):>14}  "
              f"{str(tuple(out.shape)):>22}")


# -------------------------------------------
# Exp 3 — Diferentes dimensões de embedding |
# -------------------------------------------

def exp3_dimensoes_de_embedding():
    # Varia embedding_dim e mostra o shape final e o custo em parâmetros.
    print("\n" + "=" * 60)
    print("EXP 3 — Diferentes dimensões de embedding")
    print("=" * 60)

    texto = load_text(CORPUS)
    tokenizer = create_tokenizer(texto)
    ml = 8
    bs = 4
    vocab = len(tokenizer.str_to_int)

    dl = create_dataloader_v1(
        texto, tokenizer,
        batch_size=bs, max_length=ml,
        stride=ml, shuffle=False, drop_last=True,
    )
    inputs, _ = next(iter(dl))

    print(f"{'emb_dim':>8}  {'shape_entrada':>22}  {'parametros_emb':>16}")

    for d in [16, 64, 256, 512, 768]:
        emb = GPTInputEmbeddings(vocab_size=vocab, context_length=ml, embedding_dim=d)
        out = emb(inputs)
        # parâmetros: (vocab + context_length) * d
        params = sum(p.numel() for p in emb.parameters())
        print(f"{d:>8}  {str(tuple(out.shape)):>22}  {params:>16,}")


# ------------------------------------------
# Exp 4 — Diferentes sequências de entrada |
# ------------------------------------------

def exp4_diferentes_sequencias():
    # Tokeniza frases bem distintas com o mesmo vocabulário e compara.
    print("\n" + "=" * 60)
    print("EXP 4 — Diferentes sequências de entrada")
    print("=" * 60)

    frases = [
        "Hello, world!",
        "The quick brown fox",
        "It was a dark and stormy night.",
        "She had not intended to love him.",
    ]

    texto_base = load_text(CORPUS)
    tokenizer = create_tokenizer(texto_base)
    ml = 4
    embedding_dim = 32

    print(f"{'frase':45}  {'tokens':>7}  {'ids':}")
    for frase in frases:
        try:
            ids = tokenizer.encode(frase)
        except KeyError as e:
            print(f"  {frase[:44]:45}  ERRO (token fora do vocab: {e})")
            continue
        toks = tokenize(frase)
        print(f"  {frase[:44]:45}  {len(toks):>7}  {ids[:8]}{'...' if len(ids) > 8 else ''}")

        # embedding da sequência (preenche ou corta para ml)
        ids_tensor = torch.tensor([ids[:ml]])
        if ids_tensor.shape[1] < ml:
            ids_tensor = torch.cat(
                [ids_tensor, torch.zeros(1, ml - ids_tensor.shape[1], dtype=torch.long)], dim=1
            )
        emb = GPTInputEmbeddings(
            vocab_size=len(tokenizer.str_to_int),
            context_length=ml,
            embedding_dim=embedding_dim,
        )
        out = emb(ids_tensor)
        print(f"    → embedding shape: {tuple(out.shape)}")


# -----------------------------------------------------
# Exp 5 — Quantidade de tokens para textos diferentes |
# -----------------------------------------------------

def exp5_quantidade_de_tokens():
    # Compara a tokenização (regex simples) de trechos com características distintas.
    print("\n" + "=" * 60)
    print("EXP 5 — Quantidade de tokens para textos diferentes")
    print("=" * 60)

    texto_corpus = load_text(CORPUS)
    tokenizer = create_tokenizer(texto_corpus)

    trechos = {
        "Título (1 frase)": "The Verdict",
        "Prosa narrativa (parágrafo curto)": texto_corpus[:200],
        "Corpus completo": texto_corpus,
        "Texto técnico": (
            "Tokenization splits raw text into subword units. "
            "BPE merges frequent byte pairs iteratively to build vocabulary."
        ),
        "Números e símbolos": "2024: accuracy=98.7%, loss=0.023, lr=1e-4.",
    }

    print(f"{'Trecho':35}  {'chars':>7}  {'tokens regex':>12}  {'tok/char':>9}")
    for nome, txt in trechos.items():
        toks = tokenize(txt)
        ratio = len(toks) / max(len(txt), 1)
        print(f"  {nome:33}  {len(txt):>7}  {len(toks):>12}  {ratio:>9.3f}")


# --------------------------------------------------------------------
# Exp 6 — Relação entre tamanho do contexto e quantidade de amostras |
# --------------------------------------------------------------------

def exp6_contexto_vs_amostras():
    # Mostra como aumentar max_length reduz o número de janelas disponíveis.
    print("\n" + "=" * 60)
    print("EXP 6 — Relação entre tamanho do contexto e quantidade de amostras")
    print("=" * 60)

    texto = load_text(CORPUS)
    tokenizer = create_tokenizer(texto)
    tokens = tokenize(texto)
    N = len(tokenizer.encode(texto))  # quantidade total de token IDs

    print(f"Total de token IDs no corpus: {N}")
    print(f"\n{'max_length':>12}  {'stride':>8}  {'amostras':>10}  {'cobertura%':>12}")

    for ml in [4, 8, 16, 32, 64, 128, 256]:
        for stride_ratio in [1.0]:       # stride == max_length (sem sobreposição)
            stride = ml
            ds = GPTDatasetV1(texto, tokenizer, max_length=ml, stride=stride)
            cobertura = len(ds) * ml / N * 100
            print(f"{ml:>12}  {stride:>8}  {len(ds):>10}  {cobertura:>11.1f}%")

    # mostra o efeito do stride também
    print(f"\n--- Efeito do stride com max_length=8 ---")
    print(f"{'stride':>8}  {'amostras':>10}  {'sobreposicao':>14}")
    for stride in [1, 2, 4, 8]:
        ds = GPTDatasetV1(texto, tokenizer, max_length=8, stride=stride)
        sobreposicao = max(0, 8 - stride)
        print(f"{stride:>8}  {len(ds):>10}  {sobreposicao:>14} tokens")


# -----------------------------------------------------------
# Exp 7 — Dimensões das estruturas produzidas pelo pipeline |
# -----------------------------------------------------------

def exp7_dimensoes_do_pipeline():
    # Rastreia o shape em cada etapa do pipeline para uma configuração GPT-2 small.
    print("\n" + "=" * 60)
    print("EXP 7 — Dimensões das estruturas produzidas pelo pipeline")
    print("=" * 60)

    texto = load_text(CORPUS)
    tokenizer = create_tokenizer(texto)

    # configuração próxima do GPT-2 small
    max_length = 256
    batch_size = 2
    embedding_dim = 768
    vocab_size = len(tokenizer.str_to_int)

    dl = create_dataloader_v1(
        texto, tokenizer,
        batch_size=batch_size, max_length=max_length,
        stride=max_length, shuffle=False, drop_last=True,
    )
    inputs, targets = next(iter(dl))

    torch.manual_seed(7)
    emb = GPTInputEmbeddings(
        vocab_size=vocab_size,
        context_length=max_length,
        embedding_dim=embedding_dim,
    )
    tok_emb = emb.token_embedding(inputs)
    pos_emb = emb.position_embedding(torch.arange(max_length))
    model_input = emb(inputs)

    print(f"Configuracao: vocab={vocab_size}, max_length={max_length}, "
          f"batch={batch_size}, emb_dim={embedding_dim}")
    print()
    print(f"  Etapa                       Shape                    Significado")
    print(f"  {'-'*72}")
    rows = [
        ("Token IDs (inputs)",    tuple(inputs.shape),
         "(batch, seq_len)"),
        ("Token IDs (targets)",   tuple(targets.shape),
         "(batch, seq_len)  — deslocado 1 posição"),
        ("Token embeddings",      tuple(tok_emb.shape),
         "(batch, seq_len, emb_dim)"),
        ("Positional embeddings", tuple(pos_emb.shape),
         "(seq_len, emb_dim)  — broadcast p/ batch"),
        ("Entrada do Transformer",tuple(model_input.shape),
         "(batch, seq_len, emb_dim)  — tok+pos"),
    ]
    for nome, shape, sig in rows:
        print(f"  {nome:28} {str(shape):24} {sig}")

    total_emb_params = sum(p.numel() for p in emb.parameters())
    print(f"\n  Parâmetros na camada de embedding: {total_emb_params:,}")
    print(f"  Memória aprox. (float32):          "
          f"{total_emb_params * 4 / 1024**2:.1f} MB")


# ------------------
# Ponto de entrada |
# ------------------

if __name__ == "__main__":
    mostrar_fluxo()
    mostrar_fluxo_corpus()

    exp1_tamanhos_de_contexto()
    exp2_tamanhos_de_lote()
    exp3_dimensoes_de_embedding()
    exp4_diferentes_sequencias()
    exp5_quantidade_de_tokens()
    exp6_contexto_vs_amostras()
    exp7_dimensoes_do_pipeline()