# Testes da Layer Normalization - secao 4.2

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.layer_norm import LayerNorm


def test_exemplo_a_mao_do_vetor_2_4_6():
    # media = 4, variancia = 8/3, desvio = 1.633 -> [-1.2247, 0, 1.2247]
    ln = LayerNorm(emb_dim=3)
    saida = ln(torch.tensor([[2.0, 4.0, 6.0]]))
    print("\nLayerNorm([2, 4, 6]) =", saida)
    assert torch.allclose(saida, torch.tensor([[-1.2247, 0.0, 1.2247]]), atol=1e-4)


def test_cada_token_fica_com_media_0_e_variancia_1():
    # Mesmo exemplo da secao 4.2: saida de Linear(5, 6) + ReLU.
    torch.manual_seed(123)
    lote = torch.randn(2, 5)
    camada = torch.nn.Sequential(torch.nn.Linear(5, 6), torch.nn.ReLU())
    out = camada(lote)

    ln = LayerNorm(emb_dim=6)
    out_ln = ln(out)
    media = out_ln.mean(dim=-1)
    variancia = out_ln.var(dim=-1, unbiased=False)

    print("\nMedias apos LayerNorm:", media)
    print("Variancias apos LayerNorm:", variancia)
    assert torch.allclose(media, torch.zeros(2), atol=1e-5)
    # nao da exatamente 1 por causa do eps (o livro mostra 0.9995 na versao
    # unbiased; aqui fica muito perto de 1)
    assert torch.allclose(variancia, torch.ones(2), atol=1e-3)


def test_normaliza_cada_token_independente_do_lote():
    # A LayerNorm de um token nao depende dos outros tokens nem do lote:
    # normalizar o token sozinho ou dentro de um lote da o mesmo resultado.
    torch.manual_seed(0)
    x = torch.randn(4, 3, 8)
    ln = LayerNorm(emb_dim=8)
    assert torch.allclose(ln(x)[2, 1], ln(x[2:3, 1:2])[0, 0], atol=1e-6)


def test_shape_e_parametros():
    ln = LayerNorm(emb_dim=768)
    x = torch.randn(2, 4, 768)
    assert ln(x).shape == (2, 4, 768)
    # scale (768) + shift (768)
    assert sum(p.numel() for p in ln.parameters()) == 2 * 768
