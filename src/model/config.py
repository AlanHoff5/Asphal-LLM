# Chapter 4 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - configuracoes da arquitetura GPTs

"""Dicionarios de configuracao usados para montar o GPTModel.

Chaves obrigatorias:
    vocab_size      tamanho do vocabulario (50257 = BPE do GPT-2 via tiktoken)
    context_length  maior numero de tokens que o modelo enxerga de uma vez
    emb_dim         dimensao dos embeddings (largura do "fluxo residual")
    n_heads         numero de cabecas da Multi-Head Attention
    n_layers        numero de Transformer Blocks empilhados
    drop_rate       probabilidade de dropout (embeddings, atencao e atalhos)
    qkv_bias        se as projecoes Query/Key/Value tem bias

Chaves opcionais (extensoes do projeto, usadas nos experimentos):
    ff_hidden_dim   largura da camada oculta da FeedForward (padrao 4 * emb_dim)
    tie_weights     reaproveita a matriz do Token Embedding na camada de saida
    use_shortcut    liga/desliga as conexoes residuais (padrao True)
"""

# Configuracao do GPT-2 "small" (124M) - a mesma do livro. Serve para
# conferir a contagem de parametros contra o modelo original; e grande
# demais para experimentos rapidos em CPU.
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# Configuracao reduzida do projeto: mesmo tokenizador (vocab_size) e a mesma
# estrutura, mas com embeddings, contexto e profundidade menores, para rodar
# experimentos em segundos numa CPU comum. E a base dos experimentos da
# Sprint 4 e deve ser o ponto de partida do treinamento da Sprint 5.
GPT_CONFIG_MINI = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

CHAVES_OBRIGATORIAS = (
    "vocab_size",
    "context_length",
    "emb_dim",
    "n_heads",
    "n_layers",
    "drop_rate",
    "qkv_bias",
)


def validar_config(cfg):
    """Confere se a configuracao tem tudo o que o GPTModel precisa e se as
    dimensoes sao compativeis entre si. Levanta ValueError com uma mensagem
    clara em vez de deixar o erro aparecer la dentro de um view()."""
    faltando = [chave for chave in CHAVES_OBRIGATORIAS if chave not in cfg]
    if faltando:
        raise ValueError(f"Configuracao incompleta, faltam as chaves: {faltando}")
    if cfg["emb_dim"] % cfg["n_heads"] != 0:
        raise ValueError("emb_dim deve ser divisivel por n_heads")
    if not 0.0 <= cfg["drop_rate"] < 1.0:
        raise ValueError("drop_rate deve estar no intervalo [0, 1)")
    if cfg["n_layers"] < 1:
        raise ValueError("n_layers deve ser pelo menos 1")
