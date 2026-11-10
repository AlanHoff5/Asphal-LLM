# Testes da avaliacao do modelo - secoes 5.1.2 e 5.1.3

import math
import sys
from pathlib import Path

import pytest
import tiktoken
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.config import GPT_CONFIG_MINI, criar_config
from src.model.gpt_model import GPTModel
from src.training.dados import criar_loaders_treino_validacao, dividir_treino_validacao
from src.training.dispositivo import escolher_dispositivo
from src.training.losses import (
    calc_loss_batch,
    calc_loss_loader,
    perda_chute_uniforme,
    perplexidade,
)

CFG = criar_config(GPT_CONFIG_MINI, context_length=32)


@pytest.fixture(scope="module")
def tokenizer():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def modelo():
    torch.manual_seed(123)
    return GPTModel(CFG).eval()


@pytest.fixture(scope="module")
def texto():
    return (PROJECT_ROOT / "data" / "the-verdict.txt").read_text(encoding="utf-8")


def test_cross_entropy_igual_a_conta_manual(modelo):
    # media de -log(prob do alvo) deve bater com a funcao do PyTorch
    entrada = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
    alvo = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])
    with torch.no_grad():
        probas = torch.softmax(modelo(entrada), dim=-1)
    p_alvo = torch.cat([probas[i, torch.arange(3), alvo[i]] for i in range(2)])
    manual = -torch.log(p_alvo).mean().item()
    with torch.no_grad():
        pelo_codigo = calc_loss_batch(entrada, alvo, modelo, "cpu").item()
    assert pelo_codigo == pytest.approx(manual, rel=1e-5)


def test_modelo_sem_treino_fica_perto_do_chute_uniforme(modelo):
    torch.manual_seed(0)
    entrada = torch.randint(0, CFG["vocab_size"], (2, 16))
    alvo = torch.randint(0, CFG["vocab_size"], (2, 16))
    with torch.no_grad():
        perda = calc_loss_batch(entrada, alvo, modelo, "cpu").item()
    assert abs(perda - math.log(50257)) < 1.0


def test_perplexidade_e_exponencial_da_perda():
    assert perplexidade(0.0) == pytest.approx(1.0)
    assert perplexidade(10.7940) == pytest.approx(48725.8, rel=1e-3)
    assert perplexidade(perda_chute_uniforme(50257)) == pytest.approx(50257)


def test_divisao_90_10():
    treino, val = dividir_treino_validacao("a" * 1000, 0.9)
    assert len(treino) == 900 and len(val) == 100
    with pytest.raises(ValueError):
        dividir_treino_validacao("abc", 1.5)


def test_loaders_treino_e_validacao(texto, tokenizer):
    train_loader, val_loader = criar_loaders_treino_validacao(texto, tokenizer, CFG, batch_size=4)
    x, y = next(iter(train_loader))
    assert x.shape == (4, CFG["context_length"])
    assert torch.equal(x[:, 1:], y[:, :-1])          # alvo = entrada deslocada
    assert len(train_loader) > len(val_loader) > 0


def test_texto_curto_demais_gera_erro(tokenizer):
    with pytest.raises(ValueError):
        criar_loaders_treino_validacao("texto curto " * 5, tokenizer, CFG)


def test_calc_loss_loader_respeita_num_batches(texto, tokenizer, modelo):
    train_loader, _ = criar_loaders_treino_validacao(texto, tokenizer, CFG, batch_size=4)
    perda = calc_loss_loader(train_loader, modelo, "cpu", num_batches=2)
    assert math.isfinite(perda) and perda > 0


def test_escolher_dispositivo():
    assert escolher_dispositivo("cpu").type == "cpu"
    assert escolher_dispositivo().type in {"cpu", "cuda", "xpu", "mps"}
