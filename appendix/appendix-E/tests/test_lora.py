# Testes do Apendice E - LoRA (Low-Rank Adaptation)

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.finetuning.classificador import GPTClassifier
from src.finetuning.lora import (
    LinearWithLoRA,
    LoRALayer,
    contar_parametros,
    mesclar_lora,
    parametros_lora_esperados,
    preparar_classificador_lora,
    replace_linear_with_lora,
    state_dict_lora,
)
from src.model.config import GPT_CONFIG_124M, GPT_CONFIG_MINI, criar_config
from src.pytorch_basico.rede_neural import NeuralNetwork

CFG = criar_config(GPT_CONFIG_MINI, context_length=32, n_layers=2, qkv_bias=True)


def classificador_lora(rank=4, alpha=4, seed=123):
    torch.manual_seed(seed)
    return preparar_classificador_lora(GPTClassifier(CFG, num_classes=2), rank=rank, alpha=alpha)


# ----------------
# LoRALayer      |
# ----------------


def test_lora_layer_formatos_e_b_zero():
    lora = LoRALayer(in_dim=10, out_dim=6, rank=3, alpha=3)
    assert lora.A.shape == (10, 3) and lora.B.shape == (3, 6)
    assert torch.count_nonzero(lora.B) == 0
    assert torch.equal(lora(torch.rand(2, 10)), torch.zeros(2, 6))


def test_lora_layer_escala_alpha_sobre_rank():
    torch.manual_seed(0)
    lora = LoRALayer(4, 5, rank=2, alpha=8)
    with torch.no_grad():
        lora.B.normal_()
    x = torch.rand(3, 4)
    assert torch.allclose(lora(x), 4.0 * (x @ lora.A @ lora.B))


# ----------------------
# LinearWithLoRA       |
# ----------------------


def test_linear_with_lora_comeca_igual_a_linear():
    torch.manual_seed(0)
    linear = nn.Linear(8, 4)
    camada = LinearWithLoRA(linear, rank=2, alpha=2)
    x = torch.rand(5, 8)
    assert torch.equal(camada(x), linear(x))


# -------------------------------
# replace_linear_with_lora      |
# -------------------------------


def test_replace_troca_todas_as_lineares():
    model = NeuralNetwork(2, 2)                    # 3 nn.Linear (Apendice A)
    replace_linear_with_lora(model, rank=2, alpha=2)
    assert sum(isinstance(m, LinearWithLoRA) for m in model.modules()) == 3
    # dentro do LinearWithLoRA a nn.Linear original continua existindo
    assert all(isinstance(m.linear, nn.Linear) for m in model.modules() if isinstance(m, LinearWithLoRA))


def test_replace_respeita_ignorar():
    model = classificador_lora()
    assert isinstance(model.gpt.out_head, nn.Linear)
    assert not isinstance(model.gpt.out_head, LinearWithLoRA)
    assert isinstance(model.classification_head, LinearWithLoRA)


def test_numero_de_parametros_do_livro():
    # GPT-2 124M, rank 16: 2.666.528 parametros treinaveis (Apendice E)
    cfg = dict(GPT_CONFIG_124M)
    cfg["qkv_bias"] = True
    assert parametros_lora_esperados(cfg, rank=16) == 2_666_528
    model = preparar_classificador_lora(GPTClassifier(cfg), rank=16, alpha=16)
    assert contar_parametros(model)[0] == 2_666_528


def test_so_a_e_b_sao_treinaveis():
    model = classificador_lora()
    nomes = [n for n, p in model.named_parameters() if p.requires_grad]
    assert nomes and all(n.endswith(".lora.A") or n.endswith(".lora.B") for n in nomes)
    assert contar_parametros(model)[0] == parametros_lora_esperados(CFG, rank=4)


def test_lora_nao_muda_o_modelo_antes_do_treino():
    torch.manual_seed(123)
    base = GPTClassifier(CFG, num_classes=2).eval()
    model = classificador_lora().eval()
    ids = torch.randint(0, CFG["vocab_size"], (2, 10))
    with torch.no_grad():
        assert torch.allclose(base(ids), model(ids), atol=1e-6)


# --------------------------
# Treino com LoRA          |
# --------------------------


def test_um_passo_so_mexe_no_lora():
    model = classificador_lora()
    congelados = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    b_antes = model.classification_head.lora.B.clone()

    otimizador = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    ids = torch.randint(0, CFG["vocab_size"], (4, 10))
    perda = nn.functional.cross_entropy(model(ids), torch.tensor([0, 1, 0, 1]))
    perda.backward()
    otimizador.step()

    for n, p in model.named_parameters():
        if n in congelados:
            assert torch.equal(p, congelados[n]), n
            assert p.grad is None
    assert not torch.equal(model.classification_head.lora.B, b_antes)


def test_lora_aprende_tarefa_simples():
    # spam = sequencias que comecam com o token 7, ham = com o token 3
    model = classificador_lora(rank=8, alpha=8)
    otimizador = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    torch.manual_seed(0)
    ids = torch.randint(10, 100, (16, 8))
    rotulos = torch.tensor([0, 1] * 8)
    ids[:, -1] = torch.where(rotulos == 1, 7, 3)
    perdas = []
    model.train()
    for _ in range(30):
        otimizador.zero_grad()
        perda = nn.functional.cross_entropy(model(ids), rotulos)
        perda.backward()
        otimizador.step()
        perdas.append(perda.item())
    model.eval()
    with torch.no_grad():
        acc = (model(ids).argmax(-1) == rotulos).float().mean().item()
    assert perdas[-1] < perdas[0]
    assert acc == 1.0


# ---------------------------------
# Salvar e mesclar                |
# ---------------------------------


def test_state_dict_lora_tem_so_a_e_b(tmp_path):
    model = classificador_lora()
    sd = state_dict_lora(model)
    assert sd and all(k.endswith(".lora.A") or k.endswith(".lora.B") for k in sd)
    assert sum(v.numel() for v in sd.values()) == contar_parametros(model)[0]

    # recarrega num modelo novo: strict=False porque so ha os tensores do LoRA
    arquivo = tmp_path / "lora.pth"
    torch.save(sd, arquivo)
    novo = classificador_lora(seed=999)
    faltando = novo.load_state_dict(torch.load(arquivo, weights_only=True), strict=False)
    assert not faltando.unexpected_keys
    for k, v in sd.items():
        assert torch.equal(novo.state_dict()[k], v)


def test_mesclar_da_o_mesmo_resultado():
    model = classificador_lora().eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALayer):
                m.B.normal_(std=0.02)                 # simula um LoRA ja treinado
    ids = torch.randint(0, CFG["vocab_size"], (2, 10))
    with torch.no_grad():
        antes = model(ids)
        mesclar_lora(model)
        depois = model(ids)
    assert not any(isinstance(m, LinearWithLoRA) for m in model.modules())
    assert torch.allclose(antes, depois, atol=1e-5)
