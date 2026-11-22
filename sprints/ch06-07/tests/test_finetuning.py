# Testes do módulo de fine-tuning - capítulo 6

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.config import GPT_CONFIG_MINI, criar_config
from src.finetuning.classificador import (
    GPTClassifier,
    calcular_acuracia_lote,
)
from src.finetuning.dados_cls import (
    SpamDataset,
    balancear_classes,
    dividir_treino_val_teste,
)
from src.finetuning.instrucoes import (
    InstructionDataset,
    collate_instrucoes,
    formatar_prompt,
)

CFG = criar_config(GPT_CONFIG_MINI, context_length=32)


# ----------
# Fixtures |
# ----------


@pytest.fixture(scope="module")
def tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def classificador():
    torch.manual_seed(42)
    return GPTClassifier(CFG, num_classes=2, n_blocos_treinaveis=1)


# ---------------
# GPTClassifier |
# ---------------


def test_classificador_forward_shape(classificador):
    # forward deve retornar (batch, num_classes).
    entrada = torch.randint(0, CFG["vocab_size"], (4, 16))
    with torch.no_grad():
        logits = classificador(entrada)
    assert logits.shape == (4, 2)


def test_classificador_contar_parametros(classificador):
    # contar_parametros_treinaveis retorna (treinaveis, total) coerentes.
    treinaveis, total = classificador.contar_parametros_treinaveis()
    assert 0 < treinaveis < total
    assert total > 0


def test_classificador_freeze_funcionou(classificador):
    # Parâmetros do bloco 0 devem estar congelados.
    for param in classificador.gpt.trf_blocks[0].parameters():
        assert not param.requires_grad


def test_classificador_ultimo_bloco_treinavel(classificador):
    # Último bloco deve ter gradiente habilitado.
    n = CFG["n_layers"]
    ultimo = classificador.gpt.trf_blocks[n - 1]
    assert any(p.requires_grad for p in ultimo.parameters())


def test_classificador_head_treinavel(classificador):
    # classification_head deve ser treinável.
    for param in classificador.classification_head.parameters():
        assert param.requires_grad


# ------------------------
# calcular_acuracia_lote |
# ------------------------


def test_acuracia_perfeita():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    rotulos = torch.tensor([0, 1])
    assert calcular_acuracia_lote(logits, rotulos) == pytest.approx(1.0)


def test_acuracia_zero():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    rotulos = torch.tensor([1, 0])
    assert calcular_acuracia_lote(logits, rotulos) == pytest.approx(0.0)


# ----------------------------------------------
# balancear_classes / dividir_treino_val_teste |
# ----------------------------------------------


def test_balancear_classes():
    dados = [("spam1", 1)] * 10 + [("ham1", 0)] * 100
    balanceados = balancear_classes(dados)
    n_spam = sum(1 for _, r in balanceados if r == 1)
    n_ham = sum(1 for _, r in balanceados if r == 0)
    assert n_spam == n_ham


def test_dividir_treino_val_teste_fracs():
    dados = list(range(100))
    treino, val, teste = dividir_treino_val_teste(dados, train_frac=0.7, val_frac=0.1)
    assert len(treino) == 70
    assert len(val) == 10
    assert len(teste) == 20


def test_dividir_sem_overlap():
    dados = list(range(100))
    treino, val, teste = dividir_treino_val_teste(dados, train_frac=0.7, val_frac=0.1)
    s_treino = set(treino)
    s_val = set(val)
    s_teste = set(teste)
    assert s_treino.isdisjoint(s_val)
    assert s_treino.isdisjoint(s_teste)
    assert s_val.isdisjoint(s_teste)


# -------------
# SpamDataset |
# -------------


def test_spam_dataset_tokens(tokenizer):
    dados = [("hello world", 0), ("buy now", 1)]
    ds = SpamDataset(dados, tokenizer, max_length=8, padding=True)
    ids, rotulo = ds[0]
    assert ids.shape == (8,)
    assert rotulo.item() in {0, 1}


def test_spam_dataset_padding(tokenizer):
    dados = [("hi", 0), ("hello beautiful world today", 1)]
    ds = SpamDataset(dados, tokenizer, max_length=10, padding=True)
    ids0, _ = ds[0]
    # tokens curtos devem ser preenchidos com PAD_ID=50256
    assert (ids0 == SpamDataset.PAD_ID).any()


def test_spam_dataset_trunca(tokenizer):
    texto_longo = "word " * 50
    dados = [(texto_longo, 0)]
    ds = SpamDataset(dados, tokenizer, max_length=8, padding=False)
    ids, _ = ds[0]
    assert ids.shape == (8,)


# -----------------
# formatar_prompt |
# -----------------


def test_formatar_prompt_sem_input():
    ex = {"instruction": "Diga olá", "input": "", "output": "Olá!"}
    prompt = formatar_prompt(ex)
    assert "### Instruction" in prompt
    assert "### Input" not in prompt
    assert "### Response" in prompt


def test_formatar_prompt_com_input():
    ex = {"instruction": "Traduza", "input": "Hello", "output": "Olá"}
    prompt = formatar_prompt(ex)
    assert "### Input" in prompt
    assert "Hello" in prompt


# --------------------
# InstructionDataset |
# --------------------


def test_instruction_dataset_mascaramento(tokenizer):
    # Tokens do prompt devem estar mascarados com IGNORE_INDEX=-100.
    ex = {"instruction": "Diga olá", "input": "", "output": "Olá mundo"}
    ds = InstructionDataset([ex], tokenizer, max_length=64)
    input_ids, target_ids = ds[0]
    assert (target_ids == InstructionDataset.IGNORE_INDEX).any()


def test_instruction_dataset_comprimento(tokenizer):
    ex = {"instruction": "Diga olá", "input": "", "output": "Olá"}
    ds = InstructionDataset([ex], tokenizer, max_length=32)
    input_ids, target_ids = ds[0]
    assert input_ids.shape == target_ids.shape
    assert len(input_ids) <= 32


# --------------------
# collate_instrucoes |
# --------------------


def test_collate_instrucoes_padding(tokenizer):
    # collate deve alinhar sequências de tamanhos diferentes. 
    lote = [
        (torch.tensor([1, 2, 3]), torch.tensor([2, 3, -100])),
        (torch.tensor([4, 5, 6, 7, 8]), torch.tensor([5, 6, 7, 8, -100])),
    ]
    input_batch, target_batch = collate_instrucoes(lote, pad_id=50256, device="cpu")
    assert input_batch.shape[0] == 2
    assert input_batch.shape[1] == target_batch.shape[1]
    # linha curta deve ter padding
    assert input_batch[0, -1].item() == 50256
