# Testes da inferencia e geracao autoregressiva - secao 4.7

import sys
from pathlib import Path

import pytest
import tiktoken
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.config import GPT_CONFIG_MINI, criar_config
from src.model.generation import (
    aplicar_top_k,
    generate,
    generate_text_simple,
    probabilidades_do_proximo_token,
    text_to_token_ids,
    token_ids_to_text,
)
from src.model.gpt_model import GPTModel


@pytest.fixture(scope="module")
def tokenizer():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def modelo():
    torch.manual_seed(123)
    return GPTModel(GPT_CONFIG_MINI).eval()


def test_texto_ids_e_volta(tokenizer):
    ids = text_to_token_ids("Hello, I am", tokenizer)
    assert ids.tolist() == [[15496, 11, 314, 716]]
    assert ids.shape == (1, 4)
    assert token_ids_to_text(ids, tokenizer) == "Hello, I am"


def test_gera_max_new_tokens_a_mais(modelo, tokenizer):
    ids = text_to_token_ids("Hello, I am", tokenizer)
    saida = generate_text_simple(modelo, ids, max_new_tokens=6,
                                 context_size=GPT_CONFIG_MINI["context_length"])
    print("\nTexto gerado (modelo nao treinado):", token_ids_to_text(saida, tokenizer))
    assert saida.shape == (1, 4 + 6)
    assert saida[:, :4].tolist() == ids.tolist()   # o prompt e preservado


def test_gulosa_e_deterministica(modelo, tokenizer):
    ids = text_to_token_ids("Every effort moves you", tokenizer)
    a = generate_text_simple(modelo, ids, 5, GPT_CONFIG_MINI["context_length"])
    b = generate_text_simple(modelo, ids, 5, GPT_CONFIG_MINI["context_length"])
    assert torch.equal(a, b)


def test_generate_com_temperatura_zero_igual_a_gulosa(modelo, tokenizer):
    ids = text_to_token_ids("Every effort moves you", tokenizer)
    ctx = GPT_CONFIG_MINI["context_length"]
    assert torch.equal(generate_text_simple(modelo, ids, 5, ctx),
                       generate(modelo, ids, 5, ctx, temperature=0.0))


def test_top_k_1_com_sorteio_igual_a_gulosa(modelo, tokenizer):
    ids = text_to_token_ids("Every effort moves you", tokenizer)
    ctx = GPT_CONFIG_MINI["context_length"]
    torch.manual_seed(0)
    amostrado = generate(modelo, ids, 5, ctx, temperature=1.5, top_k=1)
    assert torch.equal(amostrado, generate_text_simple(modelo, ids, 5, ctx))


def test_contexto_e_cortado_quando_passa_do_limite(tokenizer):
    # Modelo que so aceita 5 posicoes recebendo um prompt de 10 tokens:
    # sem o corte, o GPTInputEmbeddings levantaria ValueError.
    cfg = criar_config(context_length=5, emb_dim=16, n_heads=2, n_layers=1)
    torch.manual_seed(0)
    pequeno = GPTModel(cfg).eval()
    ids = torch.randint(0, cfg["vocab_size"], (1, 10))
    saida = generate_text_simple(pequeno, ids, max_new_tokens=3, context_size=5)
    assert saida.shape == (1, 13)


def test_gera_em_lote(modelo):
    ids = torch.randint(0, 50257, (3, 4))
    saida = generate_text_simple(modelo, ids, 2, GPT_CONFIG_MINI["context_length"])
    assert saida.shape == (3, 6)


def test_top_k_zera_probabilidades_fora_dos_k_maiores():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    probas = probabilidades_do_proximo_token(logits, temperature=1.0, top_k=2)
    print("\nProbabilidades com top_k=2:", probas)
    assert torch.allclose(probas, torch.tensor([[0.731, 0.269, 0.0]]), atol=1e-3)
    assert torch.isinf(aplicar_top_k(logits, 2)[0, 2])


@pytest.mark.parametrize("temperatura, esperado", [
    (1.0, [0.665, 0.245, 0.090]),
    (0.5, [0.867, 0.117, 0.016]),
    (2.0, [0.506, 0.307, 0.186]),
])
def test_temperatura_muda_a_distribuicao(temperatura, esperado):
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    probas = probabilidades_do_proximo_token(logits, temperature=temperatura)
    assert torch.allclose(probas, torch.tensor([esperado]), atol=1e-3)


def test_sorteio_so_escolhe_tokens_do_top_k(modelo, tokenizer):
    ids = text_to_token_ids("Hello, I am", tokenizer)
    with torch.no_grad():
        logits = modelo(ids)[:, -1, :]
    permitidos = set(torch.topk(logits, 5).indices[0].tolist())

    torch.manual_seed(0)
    for _ in range(20):
        saida = generate(modelo, ids, 1, GPT_CONFIG_MINI["context_length"],
                         temperature=2.0, top_k=5)
        assert saida[0, -1].item() in permitidos
