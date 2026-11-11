# Testes do laco de treinamento - secao 5.2

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
from src.training.losses import calc_loss_loader
from src.training.treino import evaluate_model, train_model_simple

CFG = criar_config(GPT_CONFIG_MINI, context_length=32, n_layers=2)


@pytest.fixture(scope="module")
def tokenizer():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def loaders(tokenizer):
    texto = (PROJECT_ROOT / "data" / "the-verdict.txt").read_text(encoding="utf-8")
    torch.manual_seed(123)
    return criar_loaders_treino_validacao(texto, tokenizer, CFG, batch_size=4)


def test_treino_diminui_a_perda(tokenizer, loaders):
    train_loader, val_loader = loaders
    torch.manual_seed(123)
    model = GPTModel(CFG)
    model.eval()
    antes = calc_loss_loader(train_loader, model, "cpu", num_batches=5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    historico = train_model_simple(model, train_loader, val_loader, optimizer, "cpu",
                                   num_epochs=1, eval_freq=10, eval_iter=2,
                                   start_context="Every effort", tokenizer=tokenizer)
    model.eval()
    depois = calc_loss_loader(train_loader, model, "cpu", num_batches=5)
    assert depois < antes - 1.0


def test_historico_tem_uma_medida_por_avaliacao(tokenizer, loaders):
    train_loader, val_loader = loaders
    torch.manual_seed(123)
    model = GPTModel(CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004)
    historico = train_model_simple(model, train_loader, val_loader, optimizer, "cpu",
                                   num_epochs=1, eval_freq=10, eval_iter=1,
                                   start_context="Every effort", tokenizer=tokenizer)
    n = len(historico["train_losses"])
    assert n == len(historico["val_losses"]) == len(historico["tokens_seen"]) == len(historico["steps"])
    assert n == (len(train_loader) - 1) // 10 + 1      # passos 0, 10, 20, ...
    assert len(historico["amostras"]) == 1             # uma amostra por epoca
    assert historico["tokens_seen"] == sorted(historico["tokens_seen"])


def test_evaluate_model_volta_para_modo_treino(loaders):
    train_loader, val_loader = loaders
    model = GPTModel(CFG).train()
    evaluate_model(model, train_loader, val_loader, "cpu", eval_iter=1)
    assert model.training


def test_avaliar_nao_muda_os_pesos(loaders):
    train_loader, val_loader = loaders
    model = GPTModel(CFG)
    antes = {k: v.clone() for k, v in model.state_dict().items()}
    evaluate_model(model, train_loader, val_loader, "cpu", eval_iter=2)
    assert all(torch.equal(antes[k], v) for k, v in model.state_dict().items())


def test_taxa_de_aprendizado_warmup_e_cosine():
    from src.training.treino import taxa_de_aprendizado
    args = dict(peak_lr=1e-3, initial_lr=1e-4, min_lr=1e-5, warmup_steps=10, total_steps=100)
    assert taxa_de_aprendizado(0, **args) == pytest.approx(1e-4)       # comeca no initial_lr
    assert taxa_de_aprendizado(10, **args) == pytest.approx(1e-3)      # pico no fim do warmup
    assert taxa_de_aprendizado(100, **args) == pytest.approx(1e-5)     # termina no min_lr
    lrs = [taxa_de_aprendizado(s, **args) for s in range(10, 101)]
    assert lrs == sorted(lrs, reverse=True)                            # so desce depois do warmup


def test_train_model_registra_um_lr_por_passo(tokenizer, loaders):
    from src.training.treino import train_model
    train_loader, val_loader = loaders
    torch.manual_seed(123)
    model = GPTModel(CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    historico = train_model(model, train_loader, val_loader, optimizer, "cpu",
                            num_epochs=1, eval_freq=10, eval_iter=1,
                            start_context="Every effort", tokenizer=tokenizer,
                            warmup_steps=5, initial_lr=1e-4, min_lr=1e-5, grad_clip=1.0)
    assert len(historico["lrs"]) == len(train_loader)
    assert max(historico["lrs"]) == pytest.approx(1e-3, rel=0.05)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
