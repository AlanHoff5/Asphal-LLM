# Chapter 4 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - Layer Normalization

# 4.2

import torch
from torch import nn


class LayerNorm(nn.Module):
    """Normaliza o vetor de CADA TOKEN para media 0 e variancia 1.

    Problema que resolve: ao passar por muitas camadas, as ativacoes podem
    crescer ou encolher demais, deixando o treino instavel (gradientes que
    explodem ou somem). A LayerNorm mantem cada vetor numa escala padrao.

        saida = scale * (x - media) / sqrt(variancia + eps) + shift

    - A media e a variancia sao calculadas na ULTIMA dimensao (dim=-1), ou
      seja, nas `emb_dim` features de um token, sem olhar os outros tokens
      nem os outros exemplos do lote. Por isso, ao contrario da Batch
      Normalization, funciona com qualquer tamanho de lote (inclusive 1,
      comum na geracao de texto).
    - `scale` e `shift` sao parametros TREINAVEIS (emb_dim valores cada).
      Comecam em 1 e 0, ou seja, no inicio nao alteram nada; o treino pode
      ajusta-los se "desnormalizar" um pouco ajudar.
    - `eps` evita divisao por zero quando a variancia e ~0.
    - `unbiased=False` divide por n (e nao n-1), como no GPT-2 original;
      isso permite carregar os pesos oficiais no Cap. 5.

    Shapes: entrada (..., emb_dim) -> saida (..., emb_dim) (nao muda).
    Parametros: 2 * emb_dim (1.536 para emb_dim=768).
    """

    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # keepdim=True deixa media/variancia com shape (..., 1), para a
        # subtracao funcionar por broadcasting em cada feature do token.
        media = x.mean(dim=-1, keepdim=True)
        variancia = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalizado = (x - media) / torch.sqrt(variancia + self.eps)
        return self.scale * x_normalizado + self.shift
