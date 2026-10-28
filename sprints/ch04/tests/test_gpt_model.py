# Testes do modelo GPT completo - secao 4.6

import sys
from pathlib import Path

import pytest
import tiktoken
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.config import GPT_CONFIG_124M, GPT_CONFIG_MINI, criar_config
from src.model.gpt_model import (
    GPTModel,
    contar_parametros,
    contar_parametros_sem_saida,
    parametros_esperados,
    tamanho_em_mb,
)


@pytest.fixture(scope="module")
def lote_do_livro():
    tokenizer = tiktoken.get_encoding("gpt2")
    frases = ["Every effort moves you", "Every day holds a"]
    return torch.stack([torch.tensor(tokenizer.encode(f)) for f in frases])


@pytest.fixture(scope="module")
def modelo_124m():
    torch.manual_seed(123)
    return GPTModel(GPT_CONFIG_124M)


def test_lote_do_livro(lote_do_livro):
    assert lote_do_livro.tolist() == [[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]]


def test_saida_sao_logits_para_todo_o_vocabulario(modelo_124m, lote_do_livro):
    logits = modelo_124m(lote_do_livro)
    print("\nShape dos logits:", tuple(logits.shape))
    assert logits.shape == (2, 4, 50257)


def test_numero_de_parametros_gpt2_small(modelo_124m):
    total = contar_parametros(modelo_124m)
    sem_saida = contar_parametros_sem_saida(modelo_124m)
    print(f"\nTotal: {total:,} | sem a camada de saida: {sem_saida:,}")
    assert total == 163_009_536
    assert sem_saida == 124_412_160


def test_matrizes_de_entrada_e_saida_tem_o_mesmo_shape(modelo_124m):
    assert modelo_124m.embeddings.token_embedding.weight.shape == (50257, 768)
    assert modelo_124m.out_head.weight.shape == (50257, 768)


def test_memoria_dos_pesos(modelo_124m):
    assert round(tamanho_em_mb(modelo_124m), 2) == 621.83


def test_weight_tying_reduz_para_124m():
    cfg = criar_config(GPT_CONFIG_124M, weight_tying=True)
    torch.manual_seed(123)
    modelo = GPTModel(cfg)
    assert modelo.out_head.weight is modelo.embeddings.token_embedding.weight
    assert contar_parametros(modelo) == 124_412_160


@pytest.mark.parametrize("cfg", [
    GPT_CONFIG_MINI,
    criar_config(emb_dim=32, n_heads=4, n_layers=3, ff_hidden_dim=48, qkv_bias=True),
    criar_config(emb_dim=32, n_heads=2, n_layers=2, weight_tying=True),
])
def test_contagem_analitica_bate_com_o_modelo(cfg):
    modelo = GPTModel(cfg)
    assert contar_parametros(modelo) == parametros_esperados(cfg)["total"]


def test_contagem_analitica_gpt2_small():
    assert parametros_esperados(GPT_CONFIG_124M)["total"] == 163_009_536


def test_contexto_maior_que_o_limite_gera_erro():
    cfg = criar_config(context_length=8)
    modelo = GPTModel(cfg)
    with pytest.raises(ValueError):
        modelo(torch.zeros(1, 9, dtype=torch.long))


def test_eval_desliga_o_dropout(lote_do_livro):
    torch.manual_seed(0)
    modelo = GPTModel(criar_config(drop_rate=0.5))

    modelo.train()
    a, b = modelo(lote_do_livro), modelo(lote_do_livro)
    assert not torch.allclose(a, b)          # dropout ativo: saidas diferentes

    modelo.eval()
    with torch.no_grad():
        a, b = modelo(lote_do_livro), modelo(lote_do_livro)
    assert torch.allclose(a, b)              # dropout desligado: deterministico
