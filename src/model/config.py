# Chapter 4 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - configuracoes da arquitetura GPT

# 4.1

"""Configuracoes (hiperparametros de arquitetura) do modelo GPT.

Todas as classes de `src/model/` recebem um dicionario `cfg` com estas
chaves. Chaves obrigatorias (as mesmas do livro):

    vocab_size      tamanho do vocabulario do tokenizador BPE (Cap. 2)
    context_length  maximo de tokens por sequencia (limite do Positional Embedding)
    emb_dim         tamanho do vetor de cada token dentro do modelo
    n_heads         numero de cabecas da Multi-Head Attention (Cap. 3)
    n_layers        numero de Transformer Blocks empilhados (NOVO no Cap. 4)
    drop_rate       taxa de dropout
    qkv_bias        se as projecoes Query/Key/Value tem bias

Chaves opcionais (extensoes do projeto, usadas nos experimentos):

    ff_hidden_dim       dimensao interna da FeedForward (padrao: 4 * emb_dim)
    drop_rate_emb       dropout apos os embeddings        (padrao: drop_rate)
    drop_rate_attn      dropout nos pesos de atencao      (padrao: drop_rate)
    drop_rate_shortcut  dropout antes de somar o atalho   (padrao: drop_rate)
    weight_tying        reusa a matriz do token embedding na camada de saida
                        (padrao: False, como no livro)

Os tres dropouts separados sao o Exercicio 4.3 do livro.
"""

# GPT-2 "small" (124M), exatamente como na secao 4.1 do livro.
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # vocabulario do tokenizador BPE do GPT-2
    "context_length": 1024,  # maximo de tokens de entrada
    "emb_dim": 768,          # dimensao dos embeddings
    "n_heads": 12,           # cabecas de atencao
    "n_layers": 12,          # Transformer Blocks
    "drop_rate": 0.1,        # dropout
    "qkv_bias": False,       # sem bias em Query/Key/Value
}

# Versao reduzida do projeto: mesma estrutura, mesmo vocabulario (tokenizador
# do GPT-2), mas com dimensoes menores para rodar em segundos numa CPU comum.
# E a configuracao usada nos experimentos da Sprint 4 e no treino da Sprint 5.
GPT_CONFIG_MINI = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# Exercicio 4.2: os outros tamanhos do GPT-2 mudam so emb_dim, n_layers e n_heads.
GPT2_TAMANHOS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def criar_config(base=None, **alteracoes):
    """Devolve uma COPIA de `base` (padrao: GPT_CONFIG_MINI) com as chaves
    alteradas. Evita modificar sem querer o dicionario global nos
    experimentos.

    Exemplo: criar_config(GPT_CONFIG_124M, n_layers=24, emb_dim=1024)
    """
    cfg = dict(GPT_CONFIG_MINI if base is None else base)
    cfg.update(alteracoes)
    return cfg
