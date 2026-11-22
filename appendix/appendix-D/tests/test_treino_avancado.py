# Testes do Apendice D - warmup, cosine decay e gradient clipping

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
from src.training.dados import criar_loaders_treino_validacao
from src.training.gradientes import find_highest_gradient, norma_do_gradiente
from src.training.losses import calc_loss_batch, calc_loss_loader
from src.training.treino import taxa_de_aprendizado, train_model

CFG = criar_config(GPT_CONFIG_MINI, context_length=32, n_layers=2)

# Valores do livro (D.1): 15 epocas, 9 lotes por epoca, 20% de warmup
PEAK, INITIAL, MIN = 0.01, 0.0001, 0.00001
TOTAL, WARMUP = 9 * 15, int(0.2 * 9 * 15)


# -----------------------------------
# D.1 + D.2 Cronograma do lr        |
# -----------------------------------


def lr(passo):
    return taxa_de_aprendizado(passo, PEAK, INITIAL, MIN, WARMUP, TOTAL)


def test_warmup_steps_do_livro():
    assert WARMUP == 27


def test_warmup_comeca_no_initial_e_chega_no_pico():
    assert lr(0) == pytest.approx(INITIAL)
    assert lr(WARMUP) == pytest.approx(PEAK)
    meio = lr(WARMUP // 2)
    assert INITIAL < meio < PEAK


def test_warmup_sobe_em_linha_reta():
    difs = [lr(s + 1) - lr(s) for s in range(WARMUP - 1)]
    assert all(d == pytest.approx(difs[0]) for d in difs)
    assert difs[0] == pytest.approx((PEAK - INITIAL) / WARMUP)


def test_cosine_desce_ate_o_minimo():
    depois = [lr(s) for s in range(WARMUP, TOTAL)]
    assert all(a >= b for a, b in zip(depois, depois[1:]))      # nunca sobe
    assert lr(TOTAL) == pytest.approx(MIN)                      # progresso = 1
    meio = WARMUP + (TOTAL - WARMUP) / 2
    assert taxa_de_aprendizado(meio, PEAK, INITIAL, MIN, WARMUP, TOTAL) == pytest.approx((PEAK + MIN) / 2)


def test_lr_sempre_entre_minimo_e_pico():
    assert all(MIN - 1e-12 <= lr(s) <= PEAK + 1e-12 for s in range(TOTAL))


# -------------------------------
# D.3 Gradient clipping         |
# -------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def loaders(tokenizer):
    texto = (PROJECT_ROOT / "data" / "the-verdict.txt").read_text(encoding="utf-8")
    torch.manual_seed(123)
    return criar_loaders_treino_validacao(texto, tokenizer, CFG, batch_size=4)


def modelo_com_gradientes(loaders, escala=1.0):
    torch.manual_seed(123)
    model = GPTModel(CFG)
    entrada, alvo = next(iter(loaders[0]))
    (calc_loss_batch(entrada, alvo, model, "cpu") * escala).backward()
    return model


def test_find_highest_gradient_sem_gradientes():
    assert find_highest_gradient(GPTModel(CFG)) is None
    assert norma_do_gradiente(GPTModel(CFG)) == 0.0


def test_find_highest_gradient_e_o_maximo(loaders):
    model = modelo_com_gradientes(loaders)
    esperado = max(p.grad.max().item() for p in model.parameters() if p.grad is not None)
    assert find_highest_gradient(model).item() == pytest.approx(esperado)


def test_norma_igual_a_do_clip_grad_norm(loaders):
    model = modelo_com_gradientes(loaders)
    norma = norma_do_gradiente(model)
    devolvida = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9).item()
    assert norma == pytest.approx(devolvida, rel=1e-5)


def test_clipping_limita_a_norma_e_mantem_a_direcao(loaders):
    # escala 100: garante uma norma bem acima de 1.0
    model = modelo_com_gradientes(loaders, escala=100.0)
    antes = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    norma_antes = norma_do_gradiente(model)
    assert norma_antes > 1.0

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert norma_do_gradiente(model) == pytest.approx(1.0, rel=1e-4)
    # todos os gradientes foram multiplicados pelo MESMO fator: direcao igual
    fator = 1.0 / norma_antes
    for n, p in model.named_parameters():
        if n in antes:
            assert torch.allclose(p.grad, antes[n] * fator, rtol=1e-4, atol=1e-9)


def test_clipping_nao_muda_norma_pequena(loaders):
    model = modelo_com_gradientes(loaders, escala=1e-4)
    antes = norma_do_gradiente(model)
    assert antes < 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert norma_do_gradiente(model) == pytest.approx(antes)


# -------------------------------
# D.4 train_model               |
# -------------------------------


def test_train_model_registra_o_cronograma_e_aprende(tokenizer, loaders):
    train_loader, val_loader = loaders
    torch.manual_seed(123)
    model = GPTModel(CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.1)
    model.eval()
    antes = calc_loss_loader(train_loader, model, "cpu", num_batches=5)

    n_epochs = 2
    total = n_epochs * len(train_loader)
    warmup = int(0.2 * total)
    historico = train_model(model, train_loader, val_loader, optimizer, "cpu",
                            num_epochs=n_epochs, eval_freq=50, eval_iter=1,
                            start_context="Every effort moves you", tokenizer=tokenizer,
                            warmup_steps=warmup, initial_lr=1e-5, min_lr=1e-5)

    lrs = historico["lrs"]
    assert len(lrs) == total
    assert lrs[0] == pytest.approx(1e-5)
    assert max(lrs) == pytest.approx(0.004)
    assert lrs.index(max(lrs)) == warmup
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.004)   # restaurado no fim

    model.eval()
    depois = calc_loss_loader(train_loader, model, "cpu", num_batches=5)
    assert depois < antes
