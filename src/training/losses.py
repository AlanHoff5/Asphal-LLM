# Chapter 5 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - perda (cross-entropy) e perplexity

# 5.1.2 e 5.1.3

import math

import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    """Cross-entropy de UM lote.

    input_batch:  (b, n)          Token IDs de entrada
    target_batch: (b, n)          o token que vem depois de cada posicao
    logits:       (b, n, vocab)   saida do GPTModel

    A cross_entropy do PyTorch espera (amostras, classes) e (amostras,),
    por isso os tensores sao "achatados":
        logits.flatten(0, 1) -> (b*n, vocab)
        target.flatten()     -> (b*n,)
    Por dentro ela faz softmax -> pega a probabilidade do alvo -> log ->
    media -> troca o sinal (o passo a passo esta em
    sprints/ch05/experiments/avaliacao_modelo.py).
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Perda MEDIA de varios lotes de um DataLoader.

    num_batches=None usa todos os lotes; um numero menor deixa a avaliacao
    mais rapida durante o treino (e o eval_iter do livro).

    Nao calcula gradientes (torch.no_grad): avaliar nao muda os pesos.
    Lembre de chamar model.eval() antes, para desligar o dropout.
    """
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total = 0.0
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            total += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total / num_batches


def perplexidade(loss):
    """Perplexity = e^perda. E o numero de tokens entre os quais o modelo esta
    "em duvida" a cada passo: quanto menor, mais certo ele esta."""
    return math.exp(loss)


def perda_chute_uniforme(vocab_size):
    """Perda de um modelo que da a MESMA probabilidade (1/vocab) a todos os
    tokens: ln(vocab). E a referencia para um modelo sem treino
    (ln(50257) = 10,82; perplexity = 50.257)."""
    return math.log(vocab_size)
