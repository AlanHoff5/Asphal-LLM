# Chapter 5 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - divisao treino/validacao e DataLoaders do treinamento

# 5.1.3

from src.data.dataloader import create_dataloader_v1


def dividir_treino_validacao(texto, train_ratio=0.90):
    """Divide o texto em duas partes, pelos CARACTERES (como no livro):
    os primeiros `train_ratio` (90%) para treino e o resto (10%) para validacao.

    A divisao e feita ANTES de tokenizar, e o trecho de validacao fica no
    final do texto. O modelo nunca aprende com ele: so e avaliado nele.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio deve ficar entre 0 e 1")
    indice = int(train_ratio * len(texto))
    return texto[:indice], texto[indice:]


def criar_loaders_treino_validacao(texto, tokenizer, cfg, batch_size=2,
                                   train_ratio=0.90, stride=None):
    """Cria os dois DataLoaders usados no treino e na avaliacao.

    - max_length = cfg["context_length"]: cada amostra tem o tamanho do contexto.
    - stride = max_length (padrao): janelas sem sobreposicao, como no livro.
    - treino: shuffle=True e drop_last=True (lotes embaralhados e completos).
    - validacao: shuffle=False e drop_last=False (sempre os mesmos lotes,
      nenhum token de validacao e descartado).

    Reaproveita o create_dataloader_v1 da Sprint 2.
    """
    contexto = cfg["context_length"]
    stride = stride or contexto
    texto_treino, texto_val = dividir_treino_validacao(texto, train_ratio)

    # Checagem do livro: cada parte precisa de mais tokens que o contexto,
    # senao o GPTDatasetV1 nao consegue montar nem uma amostra.
    for nome, parte in (("treino", texto_treino), ("validacao", texto_val)):
        n_tokens = len(tokenizer.encode(parte))
        if n_tokens <= contexto:
            raise ValueError(
                f"O conjunto de {nome} tem {n_tokens} tokens, mas o contexto e "
                f"{contexto}. Use mais texto, um context_length menor ou outro train_ratio."
            )

    train_loader = create_dataloader_v1(
        texto_treino, tokenizer, batch_size=batch_size, max_length=contexto,
        stride=stride, shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = create_dataloader_v1(
        texto_val, tokenizer, batch_size=batch_size, max_length=contexto,
        stride=stride, shuffle=False, drop_last=False, num_workers=0,
    )
    return train_loader, val_loader


def contar_tokens_do_loader(loader):
    """Quantos tokens de ENTRADA o loader entrega (soma de todos os lotes)."""
    return sum(entrada.numel() for entrada, _ in loader)
