"""Demonstra o fluxo completo de dados da Sprint 2."""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloader_v1
from src.data.dataset import GPTDatasetV1
from src.data.embeddings import GPTInputEmbeddings
from src.data.tokenizer import create_tokenizer, load_text, tokenize

# Corpus real do projeto (Seguranca da Informacao / CompTIA Security+).
# Ver data/README.md: se este arquivo nao existir no seu checkout, a
# demonstracao abaixo roda so com o texto de exemplo, e mostrar_fluxo_corpus_real()
# fica pulada. Assim que o arquivo for adicionado ao repositorio, esta
# demonstracao passa a usa-lo automaticamente, sem precisar mudar nada aqui.
CORPUS_COMPTIA = (
    PROJECT_ROOT / "data" / "comptia_security_pluse_701" / "cleaned_data.txt"
)


def mostrar_fluxo():
    texto = "um dois tres quatro cinco seis sete oito"
    max_length = 3
    stride = 1
    batch_size = 2
    embedding_dim = 5

    print("FLUXO COMPLETO DA SPRINT 2")

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

    dataset = GPTDatasetV1(
        texto,
        tokenizer,
        max_length=max_length,
        stride=stride,
    )
    primeira_entrada, primeiro_alvo = dataset[0]
    print("\nSEQUENCIAS DE TREINAMENTO")
    print("Quantidade de amostras:", len(dataset))
    print("Primeira entrada:", primeira_entrada.tolist())
    print("Primeiro alvo:", primeiro_alvo.tolist())
    print("O alvo e a entrada deslocada:", torch.equal(primeiro_alvo[:-1], primeira_entrada[1:]))

    dataloader = create_dataloader_v1(
        texto,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        drop_last=False,
    )
    inputs, targets = next(iter(dataloader))
    print("\nDATALOADER")
    print("Quantidade de lotes:", len(dataloader))
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)
    print("Shape dos inputs:", tuple(inputs.shape))
    print("Shape dos targets:", tuple(targets.shape))

    torch.manual_seed(7)
    embedding_layer = GPTInputEmbeddings(
        vocab_size=len(tokenizer.str_to_int),
        context_length=max_length,
        embedding_dim=embedding_dim,
    )
    token_embeddings = embedding_layer.token_embedding(inputs)
    positional_embeddings = embedding_layer.position_embedding(
        torch.arange(max_length)
    )
    model_input = embedding_layer(inputs)

    print("\nEMBEDDINGS")
    print("Token embeddings shape:", tuple(token_embeddings.shape))
    print("Positional embeddings shape:", tuple(positional_embeddings.shape))
    print("Entrada final do Transformer shape:", tuple(model_input.shape))
    print("Primeiro vetor da entrada final:", model_input[0, 0].tolist())


    # Demonstracao do item 3.5: a posicao altera a representacao.
    # O mesmo Token ID repetido em todas as posicoes gera vetores finais
    # diferentes, porque cada um recebe um positional embedding distinto.
    ids_repetidos = torch.tensor([[token_ids[0]] * max_length])
    saida_repetida = embedding_layer(ids_repetidos)[0]
    tok_puro = embedding_layer.token_embedding(ids_repetidos)[0]
    print("\nPOSICAO ALTERA A REPRESENTACAO (3.5)")
    print("Token ID usado em todas as posicoes:", token_ids[0])
    print("Token embedding puro igual nas posicoes 0 e 1:",
          torch.allclose(tok_puro[0], tok_puro[1]))
    print("Entrada final (token + posicao) igual nas posicoes 0 e 1:",
          torch.allclose(saida_repetida[0], saida_repetida[1]))

    print("\nFLUXO CONCLUIDO")
    print("Texto -> Tokens -> Token IDs -> Dataset -> DataLoader")
    print("-> Token Embeddings + Positional Embeddings -> Transformer")



def mostrar_fluxo_corpus_real():
    """Repete o mesmo fluxo da Sprint 2, agora sobre o corpus real do
    projeto (Seguranca da Informacao / CompTIA Security+), em vez do texto
    de exemplo. So imprime estatisticas resumidas, ja que o corpus real tem
    dezenas de milhares de tokens - dar print no vocabulario inteiro, como
    mostrar_fluxo() faz com o texto pequeno, nao seria legivel aqui.
    """
    if not CORPUS_COMPTIA.exists():
        print("\nCORPUS REAL (CompTIA Security+) AUSENTE")
        print(f"Esperado em: {CORPUS_COMPTIA}")
        print("Ver data/README.md para instrucoes de como adiciona-lo.")
        return

    max_length = 8
    stride = 8
    batch_size = 8
    embedding_dim = 256

    texto = load_text(CORPUS_COMPTIA)
    tokens = tokenize(texto)
    tokenizer = create_tokenizer(texto)

    print("\nFLUXO SOBRE O CORPUS REAL (CompTIA Security+)")
    print("Caracteres:", len(texto))
    print("Tokens:", len(tokens))
    print("Tamanho do vocabulario:", len(tokenizer.str_to_int))

    dataloader = create_dataloader_v1(
        texto,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        drop_last=True,
    )
    inputs, targets = next(iter(dataloader))
    print("Quantidade de lotes:", len(dataloader))
    print("Shape dos inputs:", tuple(inputs.shape))
    print("Shape dos targets:", tuple(targets.shape))

    torch.manual_seed(7)
    embedding_layer = GPTInputEmbeddings(
        vocab_size=len(tokenizer.str_to_int),
        context_length=max_length,
        embedding_dim=embedding_dim,
    )
    model_input = embedding_layer(inputs)
    print("Shape da entrada final do Transformer:", tuple(model_input.shape))


if __name__ == "__main__":
    mostrar_fluxo()
    mostrar_fluxo_corpus_real()
