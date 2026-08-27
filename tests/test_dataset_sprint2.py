# Testes do Dataset - item 3.3 da Sprint 2 (pares entrada-alvo)

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GPTDatasetV1
from src.data.tokenizer import create_tokenizer, load_text, tokenize


TEXTO = "um dois tres quatro cinco seis"
MAX_LENGTH = 3
STRIDE = 1


@pytest.fixture
def tokenizador():
    return create_tokenizer(TEXTO)


@pytest.fixture
def dataset(tokenizador):
    return GPTDatasetV1(TEXTO, tokenizador, MAX_LENGTH, STRIDE)


def test_dataset_produz_o_numero_esperado_de_janelas(tokenizador, dataset):
    token_ids = tokenizador.encode(TEXTO)

    print("Texto:", TEXTO)
    print("Token IDs:", token_ids)
    print("Quantidade de amostras:", len(dataset))

    esperado = len(range(0, len(token_ids) - MAX_LENGTH, STRIDE))
    assert len(dataset) == esperado


def test_cada_amostra_tem_entrada_e_alvo_com_shape_e_dtype_corretos(dataset):
    entrada, alvo = dataset[0]

    print("Entrada:", entrada.tolist(), entrada.dtype, tuple(entrada.shape))
    print("Alvo:   ", alvo.tolist(), alvo.dtype, tuple(alvo.shape))

    assert isinstance(entrada, torch.Tensor) and isinstance(alvo, torch.Tensor)
    # dtype long: os IDs serao usados como indices de nn.Embedding.
    assert entrada.dtype == torch.long and alvo.dtype == torch.long
    assert tuple(entrada.shape) == (MAX_LENGTH,)
    assert tuple(alvo.shape) == (MAX_LENGTH,)


def test_o_alvo_e_a_entrada_deslocada_uma_posicao(dataset):
    for indice in range(len(dataset)):
        entrada, alvo = dataset[indice]
        print(f"Amostra {indice}: entrada={entrada.tolist()} alvo={alvo.tolist()}")
        assert torch.equal(alvo[:-1], entrada[1:])


@pytest.mark.parametrize(
    "texto, max_length, stride",
    [
        ("um dois tres quatro cinco seis", 0, 1),   # max_length invalido
        ("um dois tres quatro cinco seis", 3, 0),   # stride invalido
        ("um dois", 3, 1),                          # texto mais curto que o contexto
    ],
)
def test_dataset_rejeita_configuracoes_invalidas(tokenizador, texto, max_length, stride):
    with pytest.raises(ValueError):
        GPTDatasetV1(texto, tokenizador, max_length, stride)


CORPUS_COMPTIA = (
    PROJECT_ROOT / "data" / "comptia_security_pluse_701" / "cleaned_data.txt"
)


@pytest.mark.skipif(
    not CORPUS_COMPTIA.exists(), reason="corpus CompTIA ausente neste checkout"
)
def test_amostra_do_corpus_real_comptia():
    texto_corpus = load_text(CORPUS_COMPTIA)
    tokenizador_corpus = create_tokenizer(texto_corpus)

    dataset_corpus = GPTDatasetV1(texto_corpus, tokenizador_corpus, max_length=8, stride=8)
    entrada, alvo = dataset_corpus[0]

    print("Tokens no corpus:", len(tokenize(texto_corpus)))
    print("Amostras no dataset:", len(dataset_corpus))
    print("Primeira entrada:", entrada.tolist())
    print("Primeiro alvo:   ", alvo.tolist())

    assert len(dataset_corpus) > 1000
    assert tuple(entrada.shape) == (8,)
    assert torch.equal(alvo[:-1], entrada[1:])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-v"]))
