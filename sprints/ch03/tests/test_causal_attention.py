# Testes de causal (masked) self-attention em lote - secao 3.5

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.attention.causal_attention import CausalAttention


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


def _pesos_de_atencao(camada, lote):
    """Recalcula os pesos de atencao internos de uma CausalAttention, para
    poder inspeciona-los diretamente nos testes (o forward() so retorna o
    vetor de contexto final)."""
    with torch.no_grad():
        queries = camada.W_query(lote)
        keys = camada.W_key(lote)
        num_tokens = lote.shape[1]
        scores = queries @ keys.transpose(1, 2)
        scores.masked_fill_(camada.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        return torch.softmax(scores / camada.d_out**0.5, dim=-1)


def test_processa_lote_de_sequencias():
    # (batch_size, num_tokens, d_in): duas copias da mesma sequencia, para
    # simular um lote como o que sai do DataLoader/GPTInputEmbeddings.
    lote = torch.stack([ENTRADAS_EXEMPLO, ENTRADAS_EXEMPLO])

    torch.manual_seed(123)
    camada = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)
    saida = camada(lote)

    print("\nShape da saida:", saida.shape)
    assert saida.shape == (2, 6, 2)


def test_mascara_causal_zera_pesos_futuros():
    # Nenhum token pode prestar atencao em um token que vem depois dele: a
    # parte estritamente acima da diagonal principal da matriz de pesos
    # precisa ser exatamente zero.
    lote = torch.stack([ENTRADAS_EXEMPLO])

    torch.manual_seed(123)
    camada = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)
    pesos = _pesos_de_atencao(camada, lote)

    parte_futura = torch.triu(pesos[0], diagonal=1)
    print("\nSoma dos pesos na parte futura (mascarada):", parte_futura.sum().item())
    assert parte_futura.sum().item() == 0.0

    # e cada linha continua somando 1 (softmax valido, mesmo com a mascara)
    for linha in pesos[0]:
        assert torch.isclose(linha.sum(), torch.tensor(1.0), atol=1e-5)


def test_primeiro_token_so_pode_ver_a_si_mesmo():
    # Consequencia direta da mascara causal: na primeira posicao da
    # sequencia so existe um token "passado" disponivel - ele mesmo. O
    # peso de atencao da posicao 0 para a posicao 0 precisa ser 1.0.
    lote = torch.stack([ENTRADAS_EXEMPLO])

    torch.manual_seed(123)
    camada = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)
    pesos = _pesos_de_atencao(camada, lote)

    print("\nPesos da primeira posicao:", pesos[0, 0])
    assert torch.isclose(pesos[0, 0, 0], torch.tensor(1.0), atol=1e-5)


def test_dropout_zera_pesos_durante_treinamento():
    # Com dropout=0.5 e o modulo em modo de treinamento (.train(), o padrao
    # ao instanciar), uma fracao dos pesos de atencao deve ser zerada.
    lote = torch.stack([ENTRADAS_EXEMPLO])

    torch.manual_seed(123)
    camada = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.5)
    camada.train()

    saida_1 = camada(lote)
    saida_2 = camada(lote)
    # com dropout ativo, duas passagens com a mesma entrada tendem a
    # produzir saidas diferentes (unidades zeradas aleatoriamente).
    print("\nSaidas iguais com dropout ativo?", torch.equal(saida_1, saida_2))
    assert not torch.equal(saida_1, saida_2)

    # em modo de avaliacao, o dropout e desligado e a saida fica deterministica.
    camada.eval()
    saida_eval_1 = camada(lote)
    saida_eval_2 = camada(lote)
    assert torch.equal(saida_eval_1, saida_eval_2)


def test_contexto_menor_que_context_length_funciona():
    # Uma sequencia mais curta que o context_length maximo configurado na
    # mascara deve funcionar normalmente (a mascara e cortada dinamicamente
    # em forward(), self.mask.bool()[:num_tokens, :num_tokens]).
    entrada_curta = ENTRADAS_EXEMPLO[:3]  # so 3 tokens
    lote = torch.stack([entrada_curta])

    torch.manual_seed(123)
    camada = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)
    saida = camada(lote)

    print("\nShape com sequencia mais curta:", saida.shape)
    assert saida.shape == (1, 3, 2)
