"""Demonstra o fluxo completo de dados da Sprint 2."""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloader_v1
from src.data.dataset import GPTDatasetV1
from src.data.embeddings import GPTInputEmbeddings
from src.data.tokenizer import create_tokenizer, tokenize


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


    print("\nFLUXO CONCLUIDO")
    print("Texto -> Tokens -> Token IDs -> Dataset -> DataLoader")
    print("-> Token Embeddings + Positional Embeddings -> Transformer")



if __name__ == "__main__":
    mostrar_fluxo()
