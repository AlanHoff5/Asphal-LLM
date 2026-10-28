# Testes da GELU e da FeedForward - secao 4.3

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.config import GPT_CONFIG_124M, criar_config
from src.model.feed_forward import GELU, FeedForward


def test_gelu_valores_de_referencia():
    gelu = GELU()
    x = torch.tensor([-1.0, -0.5, 0.0, 1.0, 3.0])
    esperado = torch.tensor([-0.159, -0.154, 0.0, 0.841, 2.996])
    saida = gelu(x)
    print("\nGELU", x.tolist(), "=", [round(v, 3) for v in saida.tolist()])
    assert torch.allclose(saida, esperado, atol=1e-3)


def test_gelu_e_proxima_da_versao_exata_do_pytorch():
    # A aproximacao com tanh deve ficar muito perto da GELU exata x * Phi(x).
    x = torch.linspace(-3, 3, 100)
    assert torch.allclose(GELU()(x), torch.nn.functional.gelu(x), atol=1e-3)


def test_gelu_nao_zera_o_gradiente_de_valores_negativos():
    # Diferenca central para a ReLU: com x negativo a ReLU tem gradiente 0;
    # a GELU tem gradiente diferente de zero.
    x_gelu = torch.tensor([-1.0], requires_grad=True)
    GELU()(x_gelu).backward()

    x_relu = torch.tensor([-1.0], requires_grad=True)
    torch.relu(x_relu).backward()

    print("\nGradiente em x=-1: GELU =", x_gelu.grad.item(), "| ReLU =", x_relu.grad.item())
    assert x_relu.grad.item() == 0.0
    assert x_gelu.grad.item() != 0.0


def test_feed_forward_mantem_o_shape():
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768)
    assert ffn(x).shape == (2, 3, 768)


def test_feed_forward_expande_4x_no_meio():
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768)
    meio = ffn.layers[0](x)
    print("\nShape no meio da FeedForward:", tuple(meio.shape))
    assert meio.shape == (2, 3, 4 * 768)


def test_feed_forward_numero_de_parametros_do_exercicio_4_1():
    ffn = FeedForward(GPT_CONFIG_124M)
    assert sum(p.numel() for p in ffn.parameters()) == 4_722_432


def test_ff_hidden_dim_opcional():
    cfg = criar_config(emb_dim=16, ff_hidden_dim=32)
    ffn = FeedForward(cfg)
    assert ffn.layers[0].out_features == 32
    assert ffn(torch.rand(1, 5, 16)).shape == (1, 5, 16)
