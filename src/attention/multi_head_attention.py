# Chapter 3 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - multi-head attention

# 3.6

import torch
from torch import nn

from src.attention.causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    """Multi-head attention "ingenua": varias CausalAttention em paralelo,
    concatenadas no final (secao 3.6.1).

    Cada "cabeca" e uma CausalAttention independente, com seu proprio
    conjunto de pesos Query/Key/Value. Cada uma aprende a prestar atencao
    em um aspecto diferente da sequencia (uma pode aprender relacoes
    sintaticas, outra relacoes semanticas, por exemplo), e os resultados de
    todas sao concatenados lado a lado no final.

    Essa versao e clara pedagogicamente, mas ineficiente: roda `num_heads`
    modulos separados em sequencia (um loop Python). A MultiHeadAttention
    abaixo (secao 3.6.2) calcula a mesma coisa com uma unica matriz de
    pesos por Query/Key/Value, processando todas as cabecas em paralelo via
    operacoes tensoriais - e a versao usada dali em diante no projeto.
    """

    def __init__(self, d_in, d_out, context_length, dropout=0.0, num_heads=2, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, entradas):
        # Cada cabeca produz um tensor (batch, num_tokens, d_out). Concatenar
        # na ultima dimensao junta as cabecas lado a lado:
        # (batch, num_tokens, d_out * num_heads).
        return torch.cat([head(entradas) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention eficiente, com pesos "divididos" (secao 3.6.2).

    Em vez de `num_heads` conjuntos separados de W_query/W_key/W_value (um
    modulo CausalAttention por cabeca), usa UMA UNICA camada nn.Linear por
    Query/Key/Value, com saida de tamanho d_out. Essa saida e entao
    reorganizada (view + transpose) em `num_heads` fatias de tamanho
    head_dim = d_out / num_heads, processadas em paralelo por operacoes de
    tensor - sem nenhum loop Python.

    E matematica e conceitualmente equivalente a MultiHeadAttentionWrapper
    (mesma ideia: varias cabecas aprendendo aspectos diferentes da
    sequencia), so que mais eficiente e o jeito como bibliotecas reais de
    LLM (incluindo o proprio GPT-2/3) implementam multi-head attention.
    Esta e a versao que sera reaproveitada no Transformer Block da Sprint 4.
    """

    def __init__(self, d_in, d_out, context_length, dropout=0.0, num_heads=2, qkv_bias=False):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out deve ser divisivel por num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        # head_dim: dimensao de cada cabeca individual, apos dividir d_out
        # igualmente entre elas.
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # out_proj combina as cabecas apos a concatenacao - uma projecao
        # linear extra que o livro tambem usa, comum na maioria das
        # implementacoes de multi-head attention (inclusive o Transformer
        # original).
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, entradas):
        batch_size, num_tokens, _ = entradas.shape

        # (batch, num_tokens, d_out)
        queries = self.W_query(entradas)
        keys = self.W_key(entradas)
        values = self.W_value(entradas)

        # Divide a ultima dimensao (d_out) em (num_heads, head_dim):
        # (batch, num_tokens, num_heads, head_dim).
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # Transpoe para trazer num_heads para antes de num_tokens:
        # (batch, num_heads, num_tokens, head_dim). Agora cada cabeca pode
        # ser tratada como um "lote extra" nas operacoes matriciais
        # seguintes, todas calculadas em paralelo.
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (batch, num_heads, num_tokens, head_dim) @ (..., head_dim, num_tokens)
        # -> (batch, num_heads, num_tokens, num_tokens): um score por par de
        # posicoes, para cada cabeca, para cada item do lote.
        scores = queries @ keys.transpose(2, 3)

        mascara_bool = self.mask.bool()[:num_tokens, :num_tokens]
        scores.masked_fill_(mascara_bool, -torch.inf)

        pesos = torch.softmax(scores / self.head_dim**0.5, dim=-1)
        pesos = self.dropout(pesos)

        # (batch, num_heads, num_tokens, num_tokens) @ (..., num_tokens, head_dim)
        # -> (batch, num_heads, num_tokens, head_dim).
        vetores_contexto = pesos @ values

        # Volta a ordem (batch, num_tokens, num_heads, head_dim) e junta as
        # cabecas de volta em uma unica dimensao d_out - o inverso do split
        # feito no inicio. .contiguous() e necessario porque .transpose()
        # so muda como o tensor e "visto" em memoria, e .view() exige um
        # layout contiguo.
        vetores_contexto = vetores_contexto.transpose(1, 2).contiguous()
        vetores_contexto = vetores_contexto.view(batch_size, num_tokens, self.d_out)

        return self.out_proj(vetores_contexto)
