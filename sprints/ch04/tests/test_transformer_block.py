# Testes das conexoes de atalho e do Transformer Block - secoes 4.4 e 4.5

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.config import GPT_CONFIG_124M, criar_config
from src.model.feed_forward import GELU
from src.model.transformer_block import TransformerBlock


class ExampleDeepNeuralNetwork(nn.Module):
    """Rede de 5 camadas, usada para medir o gradiente com e
    sem conexoes de atalho."""

    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), GELU())
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            saida = layer(x)
            if self.use_shortcut and x.shape == saida.shape:
                x = x + saida
            else:
                x = saida
        return x


def gradiente_medio_primeira_camada(use_shortcut):
    torch.manual_seed(123)
    modelo = ExampleDeepNeuralNetwork([3, 3, 3, 3, 3, 1], use_shortcut)
    saida = modelo(torch.tensor([[1.0, 0.0, -1.0]]))
    perda = nn.MSELoss()(saida, torch.tensor([[0.0]]))
    perda.backward()
    return modelo.layers[0][0].weight.grad.abs().mean().item()


def test_atalho_aumenta_o_gradiente_das_primeiras_camadas():
    sem = gradiente_medio_primeira_camada(use_shortcut=False)
    com = gradiente_medio_primeira_camada(use_shortcut=True)
    print(f"\nGradiente medio da 1a camada: sem atalho={sem:.6f} | com atalho={com:.6f}")
    # No livro: ~0.0002 sem atalho e ~0.22 com atalho.
    assert com > 100 * sem


def test_bloco_mantem_o_shape():
    torch.manual_seed(123)
    bloco = TransformerBlock(GPT_CONFIG_124M)
    x = torch.rand(2, 4, 768)
    saida = bloco(x)
    print("\nEntrada:", tuple(x.shape), "-> Saida:", tuple(saida.shape))
    assert saida.shape == x.shape


def test_bloco_numero_de_parametros_gpt2_small():
    bloco = TransformerBlock(GPT_CONFIG_124M)
    total = sum(p.numel() for p in bloco.parameters())
    atencao = sum(p.numel() for p in bloco.att.parameters())
    ff = sum(p.numel() for p in bloco.ff.parameters())
    print(f"\nBloco: total={total:,} | atencao={atencao:,} | feedforward={ff:,}")
    assert atencao == 2_360_064
    assert ff == 4_722_432
    assert total == 7_085_568


def test_bloco_e_causal():
    # Alterar um token FUTURO nao pode mudar a saida das posicoes anteriores.
    cfg = criar_config(emb_dim=16, n_heads=2, context_length=8, drop_rate=0.0)
    torch.manual_seed(0)
    bloco = TransformerBlock(cfg).eval()

    x = torch.rand(1, 6, 16)
    x_alterado = x.clone()
    x_alterado[0, 5] = torch.rand(16)  # muda so o ultimo token

    with torch.no_grad():
        a, b = bloco(x), bloco(x_alterado)
    assert torch.allclose(a[0, :5], b[0, :5], atol=1e-6)
    assert not torch.allclose(a[0, 5], b[0, 5])


def test_atalho_preserva_a_entrada_quando_as_subcamadas_sao_zeradas():
    # Se atencao e FeedForward devolverem zero, o bloco vira a identidade:
    # e o "caminho curto" que o atalho cria.
    cfg = criar_config(emb_dim=16, n_heads=2, context_length=8, drop_rate=0.0)
    bloco = TransformerBlock(cfg).eval()
    with torch.no_grad():
        for modulo in (bloco.att.out_proj, bloco.ff.layers[2]):
            modulo.weight.zero_()
            modulo.bias.zero_()
    x = torch.rand(2, 5, 16)
    assert torch.allclose(bloco(x), x)
