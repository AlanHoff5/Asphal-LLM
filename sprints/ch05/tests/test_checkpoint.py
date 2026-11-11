# Testes do salvamento e carregamento dos pesos - secao 5.4

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.config import GPT_CONFIG_MINI, criar_config
from src.model.gpt_model import GPTModel
from src.training.checkpoint import carregar_modelo, carregar_para_continuar_treino, salvar_checkpoint

CFG = criar_config(GPT_CONFIG_MINI, context_length=32, n_layers=2)


@pytest.fixture
def modelo_treinado_um_passo():
    torch.manual_seed(123)
    model = GPTModel(CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    x = torch.randint(0, CFG["vocab_size"], (2, 16))
    loss = model(x).mean()
    loss.backward()
    optimizer.step()
    return model, optimizer


def test_modelo_carregado_da_os_mesmos_logits(tmp_path, modelo_treinado_um_passo):
    model, optimizer = modelo_treinado_um_passo
    caminho = salvar_checkpoint(tmp_path / "m.pth", model, optimizer, cfg=CFG)
    carregado, checkpoint = carregar_modelo(caminho)
    x = torch.randint(0, CFG["vocab_size"], (1, 10))
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), carregado(x))
    assert checkpoint["cfg"] == CFG
    assert not carregado.training                       # volta em modo eval


def test_otimizador_restaurado(tmp_path, modelo_treinado_um_passo):
    model, optimizer = modelo_treinado_um_passo
    caminho = salvar_checkpoint(tmp_path / "m.pth", model, optimizer, cfg=CFG)
    _, opt2, _ = carregar_para_continuar_treino(caminho, lr=1e-3)
    original = optimizer.state_dict()["state"]
    restaurado = opt2.state_dict()["state"]
    assert original.keys() == restaurado.keys()
    assert torch.equal(original[0]["exp_avg"], restaurado[0]["exp_avg"])


def test_so_pesos_nao_permite_continuar(tmp_path, modelo_treinado_um_passo):
    model, _ = modelo_treinado_um_passo
    caminho = salvar_checkpoint(tmp_path / "m.pth", model, None, cfg=CFG)
    with pytest.raises(ValueError):
        carregar_para_continuar_treino(caminho)
