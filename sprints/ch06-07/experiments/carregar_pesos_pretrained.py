# Chapter 6 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM 

# Demonstra o carregamento dos pesos oficiais do GPT-2 small (124M) para
# o GPTModel do projeto, compara a geração de texto antes e depois, e salva
# o modelo carregado em disco para uso nos experimentos seguintes.

# Requer internet na primeira execução (download ~500 MB).
# Requer: pip install transformers

# Uso:
#     python sprints/ch06/experiments/carregar_pesos_pretrained.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import tiktoken

from src.model.config import GPT_CONFIG_124M
from src.model.gpt_model import GPTModel
from src.model.generation import generate, text_to_token_ids, token_ids_to_text
from src.training.dispositivo import get_device
from src.finetuning.pesos_pretrained import baixar_gpt2_hf, carregar_pesos_gpt2, verificar_mapeamento

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DEVICE     = get_device()
CHECKPOINT = PROJECT_ROOT / "data" / "gpt2_pretrained.pth"

# O GPT-2 original usa qkv_bias=True — ajustamos a config
CFG_GPT2 = dict(GPT_CONFIG_124M)
CFG_GPT2["qkv_bias"] = True

PROMPT = "Every effort moves you"

# ---------------------------------------------------------------------------
# 1. Modelo com pesos ALEATÓRIOS (referência antes do carregamento)
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. MODELO COM PESOS ALEATÓRIOS")
print("=" * 60)

torch.manual_seed(123)
modelo_aleatorio = GPTModel(CFG_GPT2).to(DEVICE)
modelo_aleatorio.eval()

tokenizer = tiktoken.get_encoding("gpt2")
encoded   = text_to_token_ids(PROMPT, tokenizer).to(DEVICE)

torch.manual_seed(42)
with torch.no_grad():
    ids = generate(modelo_aleatorio, encoded, max_new_tokens=30,
                   context_size=CFG_GPT2["context_length"],
                   temperature=1.0, top_k=50)
texto_aleatorio = token_ids_to_text(ids.cpu(), tokenizer)
print(f"Prompt : {PROMPT}")
print(f"Gerado : {texto_aleatorio}")
print()

# ---------------------------------------------------------------------------
# 2. Carrega os pesos do GPT-2 oficial (HuggingFace)
# ---------------------------------------------------------------------------

print("=" * 60)
print("2. CARREGANDO PESOS PRÉ-TREINADOS")
print("=" * 60)

hf_model = baixar_gpt2_hf("gpt2")
modelo_pretrained = GPTModel(CFG_GPT2).to(DEVICE)
carregar_pesos_gpt2(modelo_pretrained, hf_model)
modelo_pretrained.eval()

# ---------------------------------------------------------------------------
# 3. Verifica o mapeamento tensor a tensor
# ---------------------------------------------------------------------------

print()
print("3. VERIFICAÇÃO DO MAPEAMENTO")
print("-" * 40)
ok = verificar_mapeamento(modelo_pretrained, hf_model)
print(f"Resultado: {'TODOS OK ✓' if ok else 'HÁ ERROS ✗'}")
print()

# ---------------------------------------------------------------------------
# 4. Geração com pesos pré-treinados
# ---------------------------------------------------------------------------

print("=" * 60)
print("4. MODELO COM PESOS PRÉ-TREINADOS")
print("=" * 60)

torch.manual_seed(42)
with torch.no_grad():
    ids = generate(modelo_pretrained, encoded, max_new_tokens=30,
                   context_size=CFG_GPT2["context_length"],
                   temperature=1.0, top_k=50)
texto_pretrained = token_ids_to_text(ids.cpu(), tokenizer)
print(f"Prompt : {PROMPT}")
print(f"Gerado : {texto_pretrained}")
print()

# ---------------------------------------------------------------------------
# 5. Comparação direta
# ---------------------------------------------------------------------------

print("=" * 60)
print("5. COMPARAÇÃO")
print("=" * 60)
print(f"Pesos aleatórios : {texto_aleatorio}")
print(f"Pesos pré-trein. : {texto_pretrained}")
print()
print("Observação: o modelo pré-treinado gera texto em inglês fluente porque")
print("já aprendeu padrões linguísticos em 1,5 B tokens de texto da web.")
print("O modelo aleatório não sabe nada — cada token é um chute.")

# ---------------------------------------------------------------------------
# 6. Salva o modelo carregado para os próximos experimentos
# ---------------------------------------------------------------------------

torch.save({
    "model_state_dict": modelo_pretrained.state_dict(),
    "cfg": CFG_GPT2,
}, CHECKPOINT)
print()
print(f"Modelo salvo em: {CHECKPOINT}")
print(f"Tamanho: {CHECKPOINT.stat().st_size / (1024**2):.1f} MB")
