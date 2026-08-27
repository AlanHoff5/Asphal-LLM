# Chapter 2 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - DataLoader module

# 3.6

from torch.utils.data import DataLoader

from src.data.dataset import GPTDatasetV1


def create_dataloader_v1(
    texto,
    tokenizer,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # Cria um DataLoader de pares entrada-alvo para treinamento.

    # O texto e convertidos em Token IDs pelo tokenizer, organizado em janelas
    # pelo GPTDatasetV1 e agrupado em lotes pelo DataLoader do PyTorch.

    if batch_size <= 0:
        raise ValueError("batch_size deve ser positivo")
    if num_workers < 0:
        raise ValueError("num_workers nao pode ser negativo")

    dataset = GPTDatasetV1(
        texto,
        tokenizer,
        max_length=max_length,
        stride=stride,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )