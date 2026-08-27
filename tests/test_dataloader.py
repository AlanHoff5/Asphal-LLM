# Teste do Dataloader - item 3.6 da Sprint 2

# O Dataset cria janelas de Token IDs e o DataLoader agrupa essas janelas em
# lotes. 

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloader_v1
from src.data.tokenizer import create_tokenizer


@pytest.fixture
def dados():
    # O texto possui oito tokens. Com max_length=3 e stride=1, o Dataset cria
    # max_length=3 -> tokens = [um, dois, tres], stride=1, então, alvo = [dois, tres, quatro], pula de 1 em 1, por causa do stride

    # cinco janelas: os cinco possíveis pontos de início de uma sequência de
    # três tokens acompanhada do respectivo próximo token.
    texto = "um dois tres quatro cinco seis sete oito"
    tokenizer = create_tokenizer(texto)
    return texto, tokenizer


def test_dataloader_cria_lotes_de_entrada_e_alvo(dados):
    texto, tokenizer = dados

    # shuffle=False torna a saída determinística, facilitando a inspeção dos
    # Token IDs impressos e a reprodução da prova no relatório da Sprint 2.
    # batch_size=2 agrupa duas janelas por lote; drop_last=False preserva o
    # terceiro lote, que contém apenas a quinta janela disponível.
    dataloader = create_dataloader_v1(
        texto,
        tokenizer,
        batch_size=2,
        max_length=3,
        stride=1,
        shuffle=False,
        drop_last=False,
    )

    # Materializar o DataLoader permite verificar todos os lotes, inclusive o
    # último lote parcial, em vez de observar somente o primeiro lote.
    lotes = list(dataloader)
    entradas, alvos = lotes[0]

    print("\nDataLoader com drop_last=False")
    print("  Dataset:", len(dataloader.dataset), "amostras")
    print("  Lotes produzidos:", len(lotes))
    print("  Primeiro lote - entradas:\n", entradas)
    print("  Primeiro lote - alvos:\n", alvos)
    print("  Ultimo lote tem", lotes[-1][0].shape[0], "amostra(s)")
    print("  Alvos deslocados corretamente:", torch.equal(alvos[:, :-1], entradas[:, 1:]))

    # Cinco janelas, distribuídas de duas em duas, produzem três lotes quando
    # o último lote incompleto não é descartado.
    assert len(dataloader.dataset) == 5
    assert len(lotes) == 3

    # Cada lote tem o formato (batch_size, max_length), e os IDs precisam ser
    # inteiros longos porque serão usados como índices de nn.Embedding.
    assert entradas.shape == (2, 3)
    assert alvos.shape == (2, 3)
    assert entradas.dtype == torch.long
    assert alvos.dtype == torch.long

    # O alvo é a entrada deslocada uma posição: o modelo usa cada token para
    # aprender a prever o token seguinte da sequência original.
    assert torch.equal(alvos[:, :-1], entradas[:, 1:])


def test_dataloader_descarta_ultimo_lote_incompleto(dados):
    texto, tokenizer = dados

    # Com os mesmos cinco exemplos e batch_size=2, drop_last=True descarta o
    # lote que teria somente uma amostra e mantém apenas lotes uniformes.
    dataloader = create_dataloader_v1(
        texto,
        tokenizer,
        batch_size=2,
        max_length=3,
        stride=1,
        shuffle=False,
        drop_last=True,
    )

    lotes = list(dataloader)

    print("\nDataLoader com drop_last=True")
    print("  Lotes completos preservados:", len(lotes))
    print("  Shapes:", [tuple(entradas.shape) for entradas, _ in lotes])

    # Apenas dois lotes completos permanecem: 2 x 2 = 4 amostras utilizadas.
    assert len(lotes) == 2
    assert all(entradas.shape == (2, 3) for entradas, _ in lotes)


def test_dataloader_rejeita_batch_size_invalido(dados):
    texto, tokenizer = dados

    # Um lote precisa conter pelo menos uma amostra; a API deve rejeitar zero
    # antes de tentar construir o DataLoader do PyTorch.
    with pytest.raises(ValueError, match="batch_size"):
        create_dataloader_v1(texto, tokenizer, batch_size=0)

    print("\nbatch_size=0 foi rejeitado com ValueError")


if __name__ == "__main__":
    # Permite executar `python tests/test_dataloader.py` e ver os prints. No
    # uso normal, o pytest descobre os mesmos testes automaticamente.
    raise SystemExit(pytest.main([__file__, "-s", "-v"]))
