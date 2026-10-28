"""Demonstra o fluxo completo da Sprint 4 (arquitetura GPT) e roda os
experimentos pedidos no roteiro: numero de Transformer Blocks, numero de
heads, dimensao dos embeddings, dimensao da FeedForward, dropout, tamanho
do contexto e parametros de geracao (temperatura e top-k). Tambem reproduz
o experimento das conexoes de atalho (secao 4.4) e os Exercicios 4.1-4.3.

Para cada experimento sao registrados: configuracao usada, numero de
parametros, memoria dos pesos e tempo medio de um forward na CPU.

Uso (a partir da raiz do repositorio):
    python sprints/ch04/experiments/fluxo_sprint4.py

Os resultados numericos sao salvos em experiments/resultados/sprint4_resultados.json
e os graficos em experiments/resultados/*.png (requer matplotlib).
"""

from pathlib import Path
import json
import math
import sys
import time

import tiktoken
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloader_v1
from src.model.config import GPT2_TAMANHOS, GPT_CONFIG_124M, GPT_CONFIG_MINI, criar_config
from src.model.feed_forward import GELU
from src.model.generation import (
    generate,
    generate_text_simple,
    probabilidades_do_proximo_token,
    text_to_token_ids,
    token_ids_to_text,
)
from src.model.gpt_model import (
    GPTModel,
    contar_parametros,
    contar_parametros_sem_saida,
    parametros_esperados,
    tamanho_em_mb,
)

RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS = {}          # tudo que for medido vai para o JSON no final

# Lote padrao dos experimentos de custo: 4 sequencias de 64 tokens.
BATCH_SIZE = 4
SEQ_LEN = 64

torch.set_num_threads(max(1, torch.get_num_threads()))


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------

def tempo_forward_ms(modelo, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, repeticoes=7):
    """Tempo (ms) de um forward em modo de inferencia, na CPU.

    Usa o MENOR tempo entre as repeticoes (apos 2 de aquecimento): e a
    medida menos afetada por outros processos rodando na maquina. Os
    valores absolutos dependem do computador; o que importa e a tendencia.
    """
    modelo.eval()
    ids = torch.randint(0, modelo.cfg["vocab_size"], (batch_size, seq_len))
    tempos = []
    with torch.no_grad():
        modelo(ids)                                  # aquecimento
        modelo(ids)
        for _ in range(repeticoes):
            inicio = time.perf_counter()
            modelo(ids)
            tempos.append(time.perf_counter() - inicio)
    return min(tempos) * 1000


def medir(cfg, seed=123, **kwargs_tempo):
    torch.manual_seed(seed)
    modelo = GPTModel(cfg)
    partes = parametros_esperados(cfg)
    return {
        "parametros": contar_parametros(modelo),
        "parametros_blocos": partes["transformer_blocks"],
        "parametros_embeddings_e_saida": partes["token_embedding"] + partes["out_head"],
        "memoria_mb": round(tamanho_em_mb(modelo), 2),
        "forward_ms": round(tempo_forward_ms(modelo, **kwargs_tempo), 2),
    }


def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def tabela(linhas, colunas):
    larguras = [max(len(str(c)), *(len(f"{l[c]:,}" if isinstance(l[c], int) else str(l[c])) for l in linhas))
                for c in colunas]
    print("  ".join(str(c).ljust(w) for c, w in zip(colunas, larguras)))
    print("  ".join("-" * w for w in larguras))
    for l in linhas:
        print("  ".join((f"{l[c]:,}" if isinstance(l[c], int) else str(l[c])).ljust(w)
                        for c, w in zip(colunas, larguras)))


# ---------------------------------------------------------------------------
# Fluxo basico: dimensoes dos tensores dentro do modelo
# ---------------------------------------------------------------------------

def mostrar_fluxo_basico():
    titulo("FLUXO BASICO DA SPRINT 4 - DIMENSOES DOS TENSORES (GPT_CONFIG_MINI)")
    tokenizer = tiktoken.get_encoding("gpt2")
    frases = ["Every effort moves you", "Every day holds a"]
    lote = torch.stack([torch.tensor(tokenizer.encode(f)) for f in frases])
    print("Token IDs:", lote.tolist())

    torch.manual_seed(123)
    modelo = GPTModel(GPT_CONFIG_MINI).eval()

    # Forward hooks registram o shape da saida de cada componente.
    registros = []
    nomes = {
        "embeddings": "Token + Positional Embedding",
        "drop_emb": "Dropout dos embeddings",
        "trf_blocks.0.norm1": "Bloco 0 - LayerNorm 1",
        "trf_blocks.0.att": "Bloco 0 - Multi-Head Attention",
        "trf_blocks.0.norm2": "Bloco 0 - LayerNorm 2",
        "trf_blocks.0.ff.layers.0": "Bloco 0 - FeedForward Linear 1 (expande)",
        "trf_blocks.0.ff.layers.1": "Bloco 0 - FeedForward GELU",
        "trf_blocks.0.ff.layers.2": "Bloco 0 - FeedForward Linear 2 (contrai)",
        "trf_blocks.0": "Bloco 0 - saida (apos os 2 atalhos)",
        "trf_blocks": f"Saida dos {GPT_CONFIG_MINI['n_layers']} blocos",
        "final_norm": "LayerNorm final",
        "out_head": "Camada de saida -> LOGITS",
    }
    ganchos = []
    for nome, modulo in modelo.named_modules():
        if nome in nomes:
            ganchos.append(modulo.register_forward_hook(
                lambda m, e, s, nome=nome: registros.append((nomes[nome], tuple(s.shape)))))
    with torch.no_grad():
        logits = modelo(lote)
    for g in ganchos:
        g.remove()

    print(f"\n{'Componente':<44} Shape")
    print(f"{'Token IDs (entrada)':<44} {tuple(lote.shape)}")
    # os hooks disparam na ordem de termino; reordena pela ordem do dicionario
    ordem = list(nomes.values())
    vistos = {}
    for nome, shape in registros:
        vistos.setdefault(nome, shape)
    for nome in ordem:
        print(f"{nome:<44} {vistos[nome]}")

    RESULTADOS["fluxo_dimensoes_mini"] = {n: list(vistos[n]) for n in ordem}
    return logits


# ---------------------------------------------------------------------------
# Experimento 1 - conexoes de atalho (secao 4.4)
# ---------------------------------------------------------------------------

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), GELU())
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            saida = layer(x)
            x = x + saida if self.use_shortcut and x.shape == saida.shape else saida
        return x


def gradientes_por_camada(n_camadas, use_shortcut):
    torch.manual_seed(123)
    modelo = ExampleDeepNeuralNetwork([3] * n_camadas + [1], use_shortcut)
    perda = nn.MSELoss()(modelo(torch.tensor([[1.0, 0.0, -1.0]])), torch.tensor([[0.0]]))
    perda.backward()
    return [camada[0].weight.grad.abs().mean().item() for camada in modelo.layers]


def experimento_conexoes_de_atalho():
    titulo("EXPERIMENTO 1 - CONEXOES DE ATALHO E VANISHING GRADIENT (secao 4.4)")
    resultado = {}
    for n in (5, 10, 20):
        sem = gradientes_por_camada(n, use_shortcut=False)
        com = gradientes_por_camada(n, use_shortcut=True)
        resultado[n] = {"sem_atalho": sem, "com_atalho": com}
        print(f"\nRede com {n} camadas - gradiente medio |dL/dW| por camada")
        print(f"{'camada':<8}{'sem atalho':>14}{'com atalho':>14}")
        for i, (s, c) in enumerate(zip(sem, com)):
            if n <= 5 or i in (0, 1, n // 2, n - 2, n - 1):
                print(f"{i:<8}{s:>14.6f}{c:>14.6f}")
        print(f"razao ultima/primeira camada: sem atalho = {sem[-1] / sem[0]:,.1f}x | "
              f"com atalho = {com[-1] / com[0]:,.1f}x")
    RESULTADOS["atalhos"] = resultado
    return resultado


# ---------------------------------------------------------------------------
# Experimentos 2 a 5 - parametros da arquitetura
# ---------------------------------------------------------------------------

def experimento_numero_de_blocos():
    titulo("EXPERIMENTO 2 - NUMERO DE TRANSFORMER BLOCKS (n_layers)")
    linhas = []
    for n_layers in (1, 2, 4, 8, 12):
        r = medir(criar_config(n_layers=n_layers))
        linhas.append({"n_layers": n_layers, **r})
    tabela(linhas, ["n_layers", "parametros", "parametros_blocos", "memoria_mb", "forward_ms"])
    RESULTADOS["n_layers"] = linhas
    return linhas


def experimento_numero_de_heads():
    titulo("EXPERIMENTO 3 - NUMERO DE HEADS (emb_dim fixo = 128)")
    linhas = []
    for n_heads in (1, 2, 4, 8, 16):
        r = medir(criar_config(n_heads=n_heads))
        linhas.append({"n_heads": n_heads, "head_dim": 128 // n_heads, **r})
    tabela(linhas, ["n_heads", "head_dim", "parametros", "memoria_mb", "forward_ms"])
    RESULTADOS["n_heads"] = linhas
    return linhas


def experimento_dimensao_do_embedding():
    titulo("EXPERIMENTO 4 - DIMENSAO DOS EMBEDDINGS (emb_dim), n_heads = 4")
    linhas = []
    for emb_dim in (64, 128, 256, 512):
        r = medir(criar_config(emb_dim=emb_dim))
        frac = r["parametros_embeddings_e_saida"] / r["parametros"]
        linhas.append({"emb_dim": emb_dim, **r, "fracao_vocab": f"{frac:.1%}"})
    tabela(linhas, ["emb_dim", "parametros", "parametros_blocos", "fracao_vocab",
                    "memoria_mb", "forward_ms"])
    print("fracao_vocab = parcela dos parametros no token embedding + camada de saida")
    RESULTADOS["emb_dim"] = linhas
    return linhas


def experimento_dimensao_da_feedforward():
    titulo("EXPERIMENTO 5 - DIMENSAO DA FEEDFORWARD (ff_hidden_dim), emb_dim = 128")
    linhas = []
    for mult in (1, 2, 4, 8):
        cfg = criar_config(ff_hidden_dim=mult * 128)
        r = medir(cfg)
        partes = parametros_esperados(cfg)
        linhas.append({"ff_hidden_dim": f"{mult}x = {mult * 128}",
                       "ff_por_bloco": partes["feed_forward_por_bloco"],
                       "atencao_por_bloco": partes["atencao_por_bloco"], **r})
    tabela(linhas, ["ff_hidden_dim", "ff_por_bloco", "atencao_por_bloco", "parametros", "forward_ms"])
    RESULTADOS["ff_hidden_dim"] = linhas
    return linhas


# ---------------------------------------------------------------------------
# Experimento 6 - dropout
# ---------------------------------------------------------------------------

def experimento_dropout():
    titulo("EXPERIMENTO 6 - DROPOUT (modo treino x modo avaliacao)")
    tokenizer = tiktoken.get_encoding("gpt2")
    ids = text_to_token_ids("Every effort moves you", tokenizer)
    linhas = []
    for drop in (0.0, 0.1, 0.3, 0.5):
        torch.manual_seed(123)
        modelo = GPTModel(criar_config(drop_rate=drop))

        modelo.train()
        with torch.no_grad():
            x = modelo.drop_emb(modelo.embeddings(ids))
            zeros = (x == 0).float().mean().item()
            a, b = modelo(ids), modelo(ids)
        diff_treino = (a - b).abs().mean().item()

        modelo.eval()
        with torch.no_grad():
            a, b = modelo(ids), modelo(ids)
        diff_eval = (a - b).abs().mean().item()

        linhas.append({"drop_rate": drop, "zeros_apos_drop_emb": f"{zeros:.1%}",
                       "dif_logits_treino": round(diff_treino, 4),
                       "dif_logits_eval": round(diff_eval, 4)})
    tabela(linhas, ["drop_rate", "zeros_apos_drop_emb", "dif_logits_treino", "dif_logits_eval"])
    print("dif_logits = media de |logits(1a execucao) - logits(2a execucao)| para a MESMA entrada")

    # Exercicio 4.3 - tres taxas de dropout separadas
    cfg = criar_config(drop_rate_emb=0.1, drop_rate_attn=0.2, drop_rate_shortcut=0.3)
    modelo = GPTModel(cfg)
    taxas = {
        "drop_emb": modelo.drop_emb.p,
        "atencao (pesos)": modelo.trf_blocks[0].att.dropout.p,
        "antes do atalho": modelo.trf_blocks[0].drop_shortcut.p,
    }
    print("\nExercicio 4.3 - dropouts separados:", taxas)
    RESULTADOS["dropout"] = {"linhas": linhas, "exercicio_4_3": taxas}
    return linhas


# ---------------------------------------------------------------------------
# Experimento 7 - tamanho do contexto
# ---------------------------------------------------------------------------

def experimento_tamanho_do_contexto():
    titulo("EXPERIMENTO 7 - TAMANHO DO CONTEXTO")
    cfg = criar_config(context_length=512)
    torch.manual_seed(123)
    modelo = GPTModel(cfg)
    linhas = []
    for seq_len in (32, 64, 128, 256, 512):
        ms = tempo_forward_ms(modelo, batch_size=1, seq_len=seq_len, repeticoes=3)
        # elementos das matrizes de atencao: n x n por cabeca, por bloco
        elementos = seq_len * seq_len * cfg["n_heads"] * cfg["n_layers"]
        linhas.append({"num_tokens": seq_len, "forward_ms": round(ms, 2),
                       "ms_por_token": round(ms / seq_len, 3),
                       "elementos_atencao": elementos})
    tabela(linhas, ["num_tokens", "forward_ms", "ms_por_token", "elementos_atencao"])

    print("\ncontext_length (limite) x parametros do Positional Embedding:")
    pos = []
    for c in (128, 256, 512, 1024):
        p = parametros_esperados(criar_config(context_length=c))
        pos.append({"context_length": c, "pos_embedding": p["positional_embedding"],
                    "mascara_causal_buffer": c * c, "total": p["total"]})
    tabela(pos, ["context_length", "pos_embedding", "mascara_causal_buffer", "total"])
    RESULTADOS["contexto"] = {"tempo": linhas, "limite": pos}
    return linhas


# ---------------------------------------------------------------------------
# Experimento 8 - geracao: gulosa, temperatura e top-k
# ---------------------------------------------------------------------------

def entropia(probas):
    p = probas[probas > 0]
    return -(p * p.log()).sum().item()


def experimento_geracao():
    titulo("EXPERIMENTO 8 - INFERENCIA E ESTRATEGIAS DE GERACAO")
    tokenizer = tiktoken.get_encoding("gpt2")
    ctx = GPT_CONFIG_MINI["context_length"]
    torch.manual_seed(123)
    modelo = GPTModel(GPT_CONFIG_MINI).eval()

    prompt = "Hello, I am"
    ids = text_to_token_ids(prompt, tokenizer)
    print("Prompt:", repr(prompt), "->", ids.tolist())

    # Passo a passo de UMA iteracao da geracao
    with torch.no_grad():
        logits = modelo(ids)
    ultimo = logits[:, -1, :]
    probas = torch.softmax(ultimo, dim=-1)
    top = torch.topk(probas, 3)
    print("\nUma iteracao:")
    print("  logits do modelo         ", tuple(logits.shape))
    print("  logits da ultima posicao ", tuple(ultimo.shape))
    print("  soma das probabilidades  ", round(probas.sum().item(), 4))
    print("  top-3 proximos tokens    ",
          [(tokenizer.decode([i]), round(p, 6)) for p, i in zip(top.values[0].tolist(), top.indices[0].tolist())])
    print(f"  (uniforme seria 1/50257 = {1 / 50257:.6f})")

    gulosa = generate_text_simple(modelo, ids, max_new_tokens=6, context_size=ctx)
    print("\nGulosa (6 tokens):", gulosa.tolist())
    print("Texto:", repr(token_ids_to_text(gulosa, tokenizer)))

    # Efeito da temperatura e do top-k na distribuicao do proximo token
    print("\nDistribuicao do proximo token (logits da ultima posicao):")
    linhas = []
    for temperatura, top_k in ((0.1, None), (0.5, None), (1.0, None), (2.0, None),
                               (1.0, 50), (1.0, 5), (2.0, 5)):
        p = probabilidades_do_proximo_token(ultimo, temperatura, top_k)[0]
        h = entropia(p)
        linhas.append({"temperatura": temperatura, "top_k": top_k or "-",
                       "prob_max": round(p.max().item(), 5),
                       "entropia": round(h, 3),
                       "tokens_efetivos": round(math.exp(h), 1)})
    tabela(linhas, ["temperatura", "top_k", "prob_max", "entropia", "tokens_efetivos"])
    print(f"entropia maxima (uniforme) = ln(50257) = {math.log(50257):.3f}")

    # Diversidade: 10 geracoes de 6 tokens com cada estrategia
    print("\nDiversidade - 10 geracoes de 6 tokens por estrategia:")
    estrategias = [("gulosa", 0.0, None), ("T=0.5", 0.5, None), ("T=1.0", 1.0, None),
                   ("T=2.0", 2.0, None), ("T=1.0, top_k=5", 1.0, 5)]
    diversidade = []
    for nome, t, k in estrategias:
        torch.manual_seed(0)
        saidas = [tuple(generate(modelo, ids, 6, ctx, temperature=t, top_k=k)[0, 4:].tolist())
                  for _ in range(10)]
        distintas = len(set(saidas))
        tokens_unicos = len({tok for s in saidas for tok in s})
        exemplo = tokenizer.decode(list(saidas[0]))
        diversidade.append({"estrategia": nome, "sequencias_distintas": distintas,
                            "tokens_unicos": tokens_unicos, "exemplo": repr(exemplo)})
    tabela(diversidade, ["estrategia", "sequencias_distintas", "tokens_unicos", "exemplo"])

    # Custo: sem KV cache, cada passo reprocessa o contexto inteiro
    print("\nCusto da geracao autoregressiva (sem KV cache):")
    custo = []
    for n_novos in (10, 20, 40, 80):
        inicio = time.perf_counter()
        generate_text_simple(modelo, ids, max_new_tokens=n_novos, context_size=ctx)
        total = (time.perf_counter() - inicio) * 1000
        custo.append({"tokens_gerados": n_novos, "tempo_total_ms": round(total, 1),
                      "ms_por_token": round(total / n_novos, 2)})
    tabela(custo, ["tokens_gerados", "tempo_total_ms", "ms_por_token"])

    RESULTADOS["geracao"] = {
        "gulosa_ids": gulosa.tolist(),
        "gulosa_texto": token_ids_to_text(gulosa, tokenizer),
        "distribuicao": linhas, "diversidade": diversidade, "custo": custo,
    }
    return linhas


# ---------------------------------------------------------------------------
# Experimento 9 - GPT-2 124M real, weight tying e Exercicio 4.2
# ---------------------------------------------------------------------------

def experimento_gpt2_e_tamanhos():
    titulo("EXPERIMENTO 9 - GPT-2 SMALL, WEIGHT TYING E EXERCICIO 4.2")
    torch.manual_seed(123)
    modelo = GPTModel(GPT_CONFIG_124M)
    total = contar_parametros(modelo)
    print(f"GPT-2 small instanciado: {total:,} parametros")
    print(f"Descontando a camada de saida: {contar_parametros_sem_saida(modelo):,}")
    print(f"Memoria dos pesos (float32): {tamanho_em_mb(modelo):.2f} MB")
    ms = tempo_forward_ms(modelo, batch_size=1, seq_len=64, repeticoes=3)
    print(f"Forward (1 x 64 tokens) na CPU: {ms:.1f} ms")

    partes = parametros_esperados(GPT_CONFIG_124M)
    print("\nOnde estao os parametros:")
    for chave in ("token_embedding", "positional_embedding", "transformer_blocks",
                  "final_norm", "out_head"):
        print(f"  {chave:<22}{partes[chave]:>14,}  ({partes[chave] / partes['total']:.1%})")
    print(f"  por bloco: atencao = {partes['atencao_por_bloco']:,} | "
          f"FeedForward = {partes['feed_forward_por_bloco']:,} (Exercicio 4.1)")

    tying = GPTModel(criar_config(GPT_CONFIG_124M, weight_tying=True))
    print(f"\nCom weight tying: {contar_parametros(tying):,} parametros "
          f"({tamanho_em_mb(tying):.2f} MB)")
    del tying

    print("\nExercicio 4.2 - tamanhos do GPT-2 (contagem analitica, sem instanciar):")
    linhas = []
    for nome, mudancas in GPT2_TAMANHOS.items():
        cfg = criar_config(GPT_CONFIG_124M, **mudancas)
        sem_tying = parametros_esperados(cfg)["total"]
        com_tying = parametros_esperados(criar_config(cfg, weight_tying=True))["total"]
        linhas.append({"modelo": nome, **mudancas, "parametros": sem_tying,
                       "com_tying": com_tying, "memoria_mb": round(sem_tying * 4 / 1024 ** 2, 2)})
    tabela(linhas, ["modelo", "emb_dim", "n_layers", "n_heads", "parametros", "com_tying", "memoria_mb"])

    RESULTADOS["gpt2"] = {"total": total, "forward_ms_1x64": round(ms, 1),
                          "partes": partes, "tamanhos": linhas}


# ---------------------------------------------------------------------------
# Experimento 10 - integracao com o pipeline das Sprints 2 e 3
# ---------------------------------------------------------------------------

def experimento_pipeline_completo():
    titulo("EXPERIMENTO 10 - PIPELINE COMPLETO: TEXTO -> DATALOADER -> GPT -> LOGITS")
    texto = (PROJECT_ROOT / "data" / "the-verdict.txt").read_text(encoding="utf-8")
    tokenizer = tiktoken.get_encoding("gpt2")
    loader = create_dataloader_v1(texto, tokenizer, batch_size=8, max_length=64,
                                  stride=64, shuffle=False)
    entradas, alvos = next(iter(loader))

    torch.manual_seed(123)
    modelo = GPTModel(GPT_CONFIG_MINI).eval()
    with torch.no_grad():
        logits = modelo(entradas)
        # Previa do Cap. 5: a perda de entropia cruzada entre logits e alvos.
        perda = nn.functional.cross_entropy(logits.flatten(0, 1), alvos.flatten()).item()

    print("Entradas:", tuple(entradas.shape), "| Alvos:", tuple(alvos.shape))
    print("Logits:  ", tuple(logits.shape))
    print(f"Perda (cross-entropy) do modelo NAO treinado: {perda:.3f}")
    print(f"Referencia (chute uniforme): ln(50257) = {math.log(50257):.3f}")
    RESULTADOS["pipeline"] = {"entradas": list(entradas.shape), "logits": list(logits.shape),
                              "perda_inicial": round(perda, 3)}


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------

def salvar_graficos():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib nao esta instalado - pulando os graficos.")
        return
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. gradientes por camada, com e sem atalho
    fig, eixo = plt.subplots(figsize=(7, 4))
    for n, estilo in ((5, "o"), (20, "s")):
        r = RESULTADOS["atalhos"][n]
        eixo.semilogy(r["sem_atalho"], estilo + "--", label=f"{n} camadas, sem atalho")
        eixo.semilogy(r["com_atalho"], estilo + "-", label=f"{n} camadas, com atalho")
    eixo.set_xlabel("camada (0 = primeira)")
    eixo.set_ylabel("gradiente medio |dL/dW| (log)")
    eixo.set_title("Conexoes de atalho e vanishing gradient")
    eixo.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "gradientes_atalho.png", dpi=120)
    plt.close(fig)

    # 2. custo x arquitetura
    fig, eixos = plt.subplots(1, 3, figsize=(14, 4))
    l = RESULTADOS["n_layers"]
    eixos[0].plot([x["n_layers"] for x in l], [x["parametros"] / 1e6 for x in l], "o-", label="parametros (M)")
    ax2 = eixos[0].twinx()
    ax2.plot([x["n_layers"] for x in l], [x["forward_ms"] for x in l], "s--", color="tab:red")
    ax2.set_ylabel("forward (ms)", color="tab:red")
    eixos[0].set_xlabel("n_layers")
    eixos[0].set_ylabel("parametros (milhoes)")
    eixos[0].set_title("Numero de blocos")

    e = RESULTADOS["emb_dim"]
    eixos[1].bar([str(x["emb_dim"]) for x in e], [x["parametros_embeddings_e_saida"] / 1e6 for x in e],
                 label="token emb + saida")
    eixos[1].bar([str(x["emb_dim"]) for x in e], [x["parametros_blocos"] / 1e6 for x in e],
                 bottom=[x["parametros_embeddings_e_saida"] / 1e6 for x in e], label="blocos")
    eixos[1].set_xlabel("emb_dim")
    eixos[1].set_ylabel("parametros (milhoes)")
    eixos[1].set_title("Dimensao dos embeddings")
    eixos[1].legend(fontsize=8)

    c = RESULTADOS["contexto"]["tempo"]
    eixos[2].plot([x["num_tokens"] for x in c], [x["forward_ms"] for x in c], "o-")
    eixos[2].set_xlabel("tokens na entrada")
    eixos[2].set_ylabel("forward (ms)")
    eixos[2].set_title("Tamanho do contexto")
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "custo_arquitetura.png", dpi=120)
    plt.close(fig)

    # 3. temperatura: probabilidades dos 20 tokens mais provaveis
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    modelo = GPTModel(GPT_CONFIG_MINI).eval()
    with torch.no_grad():
        ultimo = modelo(text_to_token_ids("Hello, I am", tokenizer))[:, -1, :]
    ordem = torch.argsort(ultimo[0], descending=True)[:20]
    fig, eixo = plt.subplots(figsize=(8, 4))
    largura = 0.2
    for i, t in enumerate((0.1, 0.5, 1.0, 2.0)):
        p = probabilidades_do_proximo_token(ultimo, t)[0][ordem]
        eixo.bar([j + i * largura for j in range(20)], p.numpy(), largura, label=f"T={t}")
    eixo.set_yscale("log")
    eixo.set_xticks([j + 1.5 * largura for j in range(20)])
    eixo.set_xticklabels([str(j + 1) for j in range(20)])
    eixo.set_xlabel("20 tokens mais provaveis (posicao no ranking dos logits)")
    eixo.set_ylabel("probabilidade (log)")
    eixo.set_title("Efeito da temperatura - modelo nao treinado, prompt 'Hello, I am'")
    eixo.legend()
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "temperatura.png", dpi=120)
    plt.close(fig)

    print(f"\nGraficos salvos em: {RESULTADOS_DIR}")


if __name__ == "__main__":
    print("torch", torch.__version__, "| threads:", torch.get_num_threads())
    mostrar_fluxo_basico()
    experimento_conexoes_de_atalho()
    experimento_numero_de_blocos()
    experimento_numero_de_heads()
    experimento_dimensao_do_embedding()
    experimento_dimensao_da_feedforward()
    experimento_dropout()
    experimento_tamanho_do_contexto()
    experimento_geracao()
    experimento_gpt2_e_tamanhos()
    experimento_pipeline_completo()
    salvar_graficos()

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    caminho = RESULTADOS_DIR / "sprint4_resultados.json"
    caminho.write_text(json.dumps(RESULTADOS, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")
    print(f"Resultados numericos salvos em: {caminho}")
    print("\nFLUXO DA SPRINT 4 CONCLUIDO")
