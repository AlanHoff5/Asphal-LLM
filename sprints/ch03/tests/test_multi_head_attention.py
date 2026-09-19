# Testes de multi-head attention - secao 3.6

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.attention.multi_head_attention import MultiHeadAttention, MultiHeadAttentionWrapper


ENTRADAS_EXEMPLO = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)


def test_wrapper_concatena_as_cabecas():
    # MultiHeadAttentionWrapper concatena `num_heads` CausalAttention
    # independentes: a saida final tem d_out * num_heads colunas.
    lote = torch.stack([ENTRADAS_EXEMPLO])
    d_out_por_cabeca = 2
    num_heads = 3

    torch.manual_seed(123)
    camada = MultiHeadAttentionWrapper(
        d_in=3, d_out=d_out_por_cabeca, context_length=6, dropout=0.0, num_heads=num_heads
    )
    saida = camada(lote)

    print("\nShape da saida do wrapper:", saida.shape)
    assert saida.shape == (1, 6, d_out_por_cabeca * num_heads)


def test_eficiente_divide_d_out_entre_as_cabecas():
    # Na versao eficiente, d_out ja e o total (nao por cabeca): cada uma
    # das num_heads cabecas trabalha com head_dim = d_out / num_heads.
    lote = torch.stack([ENTRADAS_EXEMPLO])
    d_out = 6
    num_heads = 3

    torch.manual_seed(123)
    camada = MultiHeadAttention(d_in=3, d_out=d_out, context_length=6, dropout=0.0, num_heads=num_heads)

    assert camada.head_dim == d_out // num_heads
    saida = camada(lote)

    print("\nShape da saida (eficiente):", saida.shape)
    # a saida final tem d_out colunas (nao d_out * num_heads): as cabecas
    # sao concatenadas e depois projetadas de volta para d_out por out_proj.
    assert saida.shape == (1, 6, d_out)


def test_d_out_precisa_ser_divisivel_por_num_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_in=3, d_out=5, context_length=6, num_heads=2)


def test_processa_lote_com_mais_de_uma_sequencia():
    lote = torch.stack([ENTRADAS_EXEMPLO, ENTRADAS_EXEMPLO, ENTRADAS_EXEMPLO])

    torch.manual_seed(123)
    camada = MultiHeadAttention(d_in=3, d_out=4, context_length=6, dropout=0.0, num_heads=2)
    saida = camada(lote)

    print("\nShape com batch_size=3:", saida.shape)
    assert saida.shape == (3, 6, 4)


def test_mascara_causal_tambem_vale_para_cada_cabeca():
    # Mesma verificacao da CausalAttention (Sprint 3, secao 3.5), agora
    # para a MultiHeadAttention: nenhuma cabeca pode "ver" o futuro.
    lote = torch.stack([ENTRADAS_EXEMPLO])
    num_heads = 2

    torch.manual_seed(123)
    camada = MultiHeadAttention(d_in=3, d_out=4, context_length=6, dropout=0.0, num_heads=num_heads)

    with torch.no_grad():
        b, n, _ = lote.shape
        q = camada.W_query(lote).view(b, n, camada.num_heads, camada.head_dim).transpose(1, 2)
        k = camada.W_key(lote).view(b, n, camada.num_heads, camada.head_dim).transpose(1, 2)
        scores = q @ k.transpose(2, 3)
        scores.masked_fill_(camada.mask.bool()[:n, :n], -torch.inf)
        pesos = torch.softmax(scores / camada.head_dim**0.5, dim=-1)

    for cabeca in range(num_heads):
        parte_futura = torch.triu(pesos[0, cabeca], diagonal=1)
        assert parte_futura.sum().item() == 0.0


def test_numero_de_parametros_e_o_esperado():
    # d_in=8, d_out=8: 3 camadas Linear (Q, K, V) sem bias de 8x8 = 3*64,
    # mais a projecao de saida out_proj (8x8 + bias de 8) = 64 + 8.
    torch.manual_seed(0)
    camada = MultiHeadAttention(d_in=8, d_out=8, context_length=4, num_heads=2, qkv_bias=False)

    total_params = sum(p.numel() for p in camada.parameters())
    esperado = 3 * (8 * 8) + (8 * 8 + 8)
    print("\nTotal de parametros:", total_params, "esperado:", esperado)
    assert total_params == esperado
