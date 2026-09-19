# Chapter 3 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - self-attention (com e sem pesos treinaveis)

# 3.3 & 3.4

import torch
from torch import nn


def attention_scores_simples(entradas):
    """Autoatencao SEM pesos treinaveis (secao 3.3 do livro).

    Serve so para demonstrar a ideia central antes de introduzir Query,
    Key e Value: o "attention score" entre dois tokens e o produto escalar
    (dot product) dos seus proprios vetores de entrada, e o "vetor de
    contexto" de um token e a media ponderada de todos os vetores de
    entrada, usando esses scores normalizados (softmax) como peso.

    `entradas` tem shape (context_length, embedding_dim): uma unica
    sequencia (sem dimensao de batch), como no exemplo do livro.

    Retorna (attention_weights, context_vectors):
      attention_weights: shape (context_length, context_length), cada
        linha soma 1.0 (e uma distribuicao de probabilidade sobre os
        tokens da sequencia).
      context_vectors: shape (context_length, embedding_dim).
    """
    # scores[i, j] = produto escalar entre o token i e o token j: quanto
    # mais alinhados (similares) os dois vetores, maior o score.
    scores = entradas @ entradas.T

    # softmax normaliza cada linha para somar 1, transformando os scores
    # em pesos de atencao (Attention Weights) interpretaveis como "quanto
    # o token i presta atencao em cada token j".
    pesos = torch.softmax(scores, dim=-1)

    # O vetor de contexto de cada token e a soma dos vetores de entrada de
    # toda a sequencia, ponderada pelos pesos de atencao daquele token.
    vetores_contexto = pesos @ entradas

    return pesos, vetores_contexto


class SelfAttention_v1(nn.Module):
    """Autoatencao COM pesos treinaveis, usando nn.Parameter (secao 3.4.1).

    Introduz Query, Key e Value: em vez de usar o vetor de entrada
    diretamente (como em attention_scores_simples), cada token e projetado
    em tres papeis diferentes por matrizes de pesos treinaveis:
      - Query (W_query): "o que este token esta procurando".
      - Key (W_key): "o que este token oferece, para ser encontrado".
      - Value (W_value): "o conteudo que este token entrega" ao vetor de
        contexto, uma vez que sua relevancia (Key) foi avaliada.

    Esta e a versao "didatica", com nn.Parameter cru (matmul explicito),
    igual ao livro. A versao usada no restante do projeto e a
    SelfAttention_v2 (com nn.Linear), que inicializa os pesos de forma mais
    estavel e e o padrao usado pelo PyTorch.
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # d_in: dimensao do vetor de entrada (embedding_dim do Capitulo 2).
        # d_out: dimensao dos vetores Query/Key/Value produzidos.
        self.d_out = d_out

        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

        if qkv_bias:
            self.b_query = nn.Parameter(torch.zeros(d_out))
            self.b_key = nn.Parameter(torch.zeros(d_out))
            self.b_value = nn.Parameter(torch.zeros(d_out))
        else:
            self.b_query = self.b_key = self.b_value = 0.0

    def forward(self, entradas):
        # entradas: shape (context_length, d_in).
        queries = entradas @ self.W_query + self.b_query
        keys = entradas @ self.W_key + self.b_key
        values = entradas @ self.W_value + self.b_value

        # Scaled Dot-Product Attention: o produto escalar Query x Key^T
        # cresce em magnitude com a dimensao d_out (mais termos somados).
        # Em dimensoes altas, scores muito grandes fazem o softmax se
        # comportar quase como uma funcao degrau (um peso vira ~1 e os
        # outros ~0), o que deixa os gradientes proximos de zero durante o
        # treinamento. Dividir por sqrt(d_out) mantem a variancia dos
        # scores estavel independente da dimensao escolhida.
        scores = queries @ keys.T
        pesos = torch.softmax(scores / self.d_out**0.5, dim=-1)

        vetores_contexto = pesos @ values
        return vetores_contexto


class SelfAttention_v2(nn.Module):
    """Mesma ideia da SelfAttention_v1, mas com nn.Linear (secao 3.4.2).

    nn.Linear usa uma inicializacao de pesos (Kaiming/He) mais adequada
    para treinamento estavel do que torch.rand puro, e e o jeito idiomatico
    de declarar uma camada linear em PyTorch. Matematicamente calcula a
    mesma coisa que a v1: y = x @ W^T + b.
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, entradas):
        queries = self.W_query(entradas)
        keys = self.W_key(entradas)
        values = self.W_value(entradas)

        scores = queries @ keys.transpose(-2, -1)
        pesos = torch.softmax(scores / self.d_out**0.5, dim=-1)

        vetores_contexto = pesos @ values
        return vetores_contexto
