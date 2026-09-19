# Chapter 3 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - causal (masked) self-attention, em lote

# 3.5

import torch
from torch import nn


class CausalAttention(nn.Module):
    """Self-attention causal, com dropout, operando em lotes (secao 3.5.2).

    Duas diferencas em relacao a SelfAttention_v2:

    1) Mascara causal: para gerar texto token a token (Next-Word
       Prediction), o token na posicao i so pode "ver" os tokens de 0 a i,
       nunca os que vem depois dele - senao o modelo estaria "espiando" a
       resposta durante o treinamento. Isso e implementado zerando, antes
       do softmax, todas as posicoes acima da diagonal principal da matriz
       de scores (substituindo por -infinito, para que o softmax as
       transforme em peso 0).

    2) Dropout: zera aleatoriamente uma fracao dos pesos de atencao durante
       o treinamento (nao durante avaliacao/inferencia), o que evita que o
       modelo dependa demais de uma unica conexao token-a-token especifica
       (overfitting).

    Tambem passa a operar sobre um LOTE de sequencias de uma vez, shape
    (batch_size, num_tokens, d_in) - o mesmo formato que sai do
    GPTInputEmbeddings do Capitulo 2 - em vez de uma sequencia isolada como
    a SelfAttention_v1/v2 do inicio do capitulo.
    """

    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # register_buffer: nao e um parametro treinavel, mas deve viajar
        # junto com o modulo quando ele for movido para outro `device`
        # (.to("cuda"), por exemplo) e ser salvo/carregado com o modelo.
        # torch.triu com diagonal=1 marca com 1 tudo ACIMA da diagonal
        # principal - exatamente as posicoes futuras que devem ser
        # mascaradas.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, entradas):
        # entradas: shape (batch_size, num_tokens, d_in).
        batch_size, num_tokens, _ = entradas.shape

        queries = self.W_query(entradas)
        keys = self.W_key(entradas)
        values = self.W_value(entradas)

        # (batch_size, num_tokens, d_out) @ (batch_size, d_out, num_tokens)
        # -> (batch_size, num_tokens, num_tokens): um score para cada par
        # de posicoes, calculado em paralelo para todo o lote.
        scores = queries @ keys.transpose(1, 2)

        # Aplica a mascara causal: onde mask == 1 (posicoes futuras), o
        # score vira -infinito. masked_fill_ e feito com a mascara ja
        # cortada para o tamanho real da sequencia (pode ser menor que
        # context_length), e convertida para bool.
        scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        # softmax(-inf) = 0, entao as posicoes futuras ficam com peso
        # exatamente zero - nenhuma informacao "vaza" do futuro.
        pesos = torch.softmax(scores / self.d_out**0.5, dim=-1)
        pesos = self.dropout(pesos)

        vetores_contexto = pesos @ values
        return vetores_contexto
