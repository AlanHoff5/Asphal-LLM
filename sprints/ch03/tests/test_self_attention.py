# Testes de self-attention (com e sem pesos treinaveis) - secoes 3.3 e 3.4

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.attention.self_attention import (
    SelfAttention_v1,
    SelfAttention_v2,
    attention_scores_simples,
)


# Mesmas entradas de exemplo do livro (embeddings de
# "Your journey starts with one step"), para poder comparar com os valores
# de referencia publicados no material do capitulo.
ENTRADAS_EXEMPLO = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ]
)


def test_pesos_de_atencao_somam_um():
    # Cada linha da matriz de pesos e uma distribuicao de probabilidade
    # sobre os tokens da sequencia: precisa somar exatamente 1.
    pesos, _ = attention_scores_simples(ENTRADAS_EXEMPLO)

    print("\nPesos de atencao (sem pesos treinaveis):")
    print(pesos)
    assert pesos.shape == (6, 6)
    for linha in pesos:
        assert torch.isclose(linha.sum(), torch.tensor(1.0), atol=1e-5)


def test_vetor_de_contexto_bate_com_referencia_do_livro():
    # Valor de referencia do livro (Chapter 3) para o vetor de contexto do
    # token "journey" (indice 1), calculado sem pesos treinaveis.
    _, vetores_contexto = attention_scores_simples(ENTRADAS_EXEMPLO)

    esperado = torch.tensor([0.4419, 0.6515, 0.5683])
    print("\nVetor de contexto de 'journey':", vetores_contexto[1])
    assert torch.allclose(vetores_contexto[1], esperado, atol=1e-3)


def test_self_attention_v1_produz_shape_correto():
    torch.manual_seed(123)
    camada = SelfAttention_v1(d_in=3, d_out=2)
    saida = camada(ENTRADAS_EXEMPLO)

    print("\nSaida da SelfAttention_v1:")
    print(saida)
    assert saida.shape == (6, 2)


def test_self_attention_v1_e_v2_tem_parametros_treinaveis():
    # W_query, W_key e W_value devem exigir gradiente: e assim que o
    # treinamento (fora do escopo desta Sprint) vai ajusta-los.
    v1 = SelfAttention_v1(d_in=3, d_out=2)
    assert v1.W_query.requires_grad
    assert v1.W_key.requires_grad
    assert v1.W_value.requires_grad

    v2 = SelfAttention_v2(d_in=3, d_out=2)
    parametros_v2 = list(v2.parameters())
    assert len(parametros_v2) > 0
    assert all(p.requires_grad for p in parametros_v2)


def test_self_attention_v2_produz_shape_correto():
    torch.manual_seed(789)
    camada = SelfAttention_v2(d_in=3, d_out=2)
    saida = camada(ENTRADAS_EXEMPLO)

    print("\nSaida da SelfAttention_v2:")
    print(saida)
    assert saida.shape == (6, 2)


def test_escalonamento_evita_softmax_saturado_em_dimensao_alta():
    # Sem dividir por sqrt(d_out), scores em dimensoes altas tendem a ficar
    # grandes demais e o softmax se comporta quase como uma funcao degrau
    # (um peso ~1, os demais ~0), o que deixa os gradientes proximos de
    # zero durante o treinamento. Aqui comparamos, para os MESMOS scores
    # brutos (fixos, simulando uma dimensao alta), o softmax COM e SEM a
    # divisao por sqrt(d_out).
    d_out = 100
    torch.manual_seed(0)
    scores_brutos = torch.randn(6, 6) * d_out**0.5  # magnitude tipica de d_out=100

    pesos_sem_escala = torch.softmax(scores_brutos, dim=-1)
    pesos_com_escala = torch.softmax(scores_brutos / d_out**0.5, dim=-1)

    maior_peso_sem_escala = pesos_sem_escala.max(dim=-1).values.mean().item()
    maior_peso_com_escala = pesos_com_escala.max(dim=-1).values.mean().item()

    print("\nMaior peso medio por linha, SEM escala:", maior_peso_sem_escala)
    print("Maior peso medio por linha, COM escala:", maior_peso_com_escala)

    # sem a escala, o softmax fica bem mais "saturado" (concentrado em um
    # unico token) do que com a escala.
    assert maior_peso_sem_escala > maior_peso_com_escala
