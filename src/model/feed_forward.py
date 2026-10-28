# Chapter 4 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - GELU e rede FeedForward

# 4.3

import math

import torch
from torch import nn


class GELU(nn.Module):
    """Gaussian Error Linear Unit, na aproximacao com tanh usada no GPT-2.

        GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))

    A versao exata e x * Phi(x) (Phi = CDF da normal padrao); a aproximacao
    e mais barata e foi a usada no treino do GPT-2.

    Diferenca para a ReLU (max(0, x)): a GELU e suave em todo ponto e, para
    x negativo, devolve um valor pequeno e negativo em vez de 0 (minimo
    ~ -0.17 em x ~ -0.75). Assim o gradiente para entradas negativas nao e
    zerado e o neuronio continua aprendendo.
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Rede FeedForward aplicada a CADA TOKEN separadamente.

        Linear(emb_dim -> hidden) -> GELU -> Linear(hidden -> emb_dim)

    Por padrao hidden = 4 * emb_dim (3072 para o GPT-2 small), como no livro.
    A chave opcional `ff_hidden_dim` do cfg permite variar esse tamanho
    (experimento da Sprint 4).

    - EXPANDE e depois CONTRAI: o espaco maior no meio permite combinar as
      features do token de forma nao linear; voltar para emb_dim e
      obrigatorio para somar o atalho (conexao residual) e para o proximo
      bloco receber o mesmo shape.
    - Atencao x FeedForward: a atencao MISTURA informacao ENTRE os tokens;
      a FeedForward PROCESSA a informacao DENTRO de cada token (os mesmos
      pesos sao aplicados em todas as posicoes, sem olhar as vizinhas).

    Shapes (emb_dim=768): (b, n, 768) -> (b, n, 3072) -> GELU -> (b, n, 768).
    Parametros: emb*hidden + hidden + hidden*emb + emb (4.722.432 no GPT-2 small).
    """

    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        hidden_dim = cfg.get("ff_hidden_dim", 4 * emb_dim)
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),   # expande
            GELU(),                           # nao linearidade
            nn.Linear(hidden_dim, emb_dim),   # contrai de volta
        )

    def forward(self, x):
        return self.layers(x)
