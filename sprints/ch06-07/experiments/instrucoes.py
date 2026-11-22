# Chapter 7 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM 

# Fine-tuning do GPT-2 pré-treinado para seguir instruções em linguagem natural.

# Requer que `carregar_pesos_pretrained.py` já tenha rodado e gerado
#   data/gpt2_pretrained.pth

# Requer internet na primeira execução (download do dataset de instruções ~180 KB).

# Uso:
#     python sprints/ch06/experiments/instrucoes.py

# Saída:
#    sprints/ch06/experiments/resultados/instrucoes_resultados.json


import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import tiktoken

from src.model.config import GPT_CONFIG_124M
from src.model.gpt_model import GPTModel
from src.model.generation import generate, text_to_token_ids, token_ids_to_text
from src.training.dispositivo import get_device
from src.finetuning.instrucoes import (
    baixar_dataset_instrucoes,
    carregar_instrucoes,
    formatar_prompt,
    criar_loaders_instrucoes,
    treinar_instrucoes,
)

# --------------
# Configuração |
# --------------

DEVICE     = get_device()
DATA_DIR   = PROJECT_ROOT / "data"
CHECKPOINT = DATA_DIR / "gpt2_pretrained.pth"
RESULTADO  = Path(__file__).parent / "resultados" / "instrucoes_resultados.json"
RESULTADO.parent.mkdir(parents=True, exist_ok=True)

CFG_GPT2 = dict(GPT_CONFIG_124M)
CFG_GPT2["qkv_bias"] = True

BATCH_SIZE    = 4
NUM_EPOCHS    = 2       # 2 épocas já mostram melhora visível; mais = mais tempo
LEARNING_RATE = 5e-5
MAX_LENGTH    = 512     # trunca prompts longos

# --------------------------
# 1. Dataset de instruções |
# --------------------------

print("=" * 60)
print("1. PREPARANDO O DATASET DE INSTRUÇÕES")
print("=" * 60)

tokenizer   = tiktoken.get_encoding("gpt2")
arquivo_json = baixar_dataset_instrucoes(DATA_DIR)
dados        = carregar_instrucoes(arquivo_json)

# Divisão: 85% treino, 15% validação (sem teste separado — avaliação qualitativa)
n_treino = int(0.85 * len(dados))
dados_treino = dados[:n_treino]
dados_val    = dados[n_treino:]

print(f"Total de exemplos : {len(dados)}")
print(f"Treino            : {len(dados_treino)}")
print(f"Validação         : {len(dados_val)}")
print()

# Mostra 2 exemplos do dataset para entender o formato
print("Exemplo de prompt formatado:")
print("-" * 40)
print(formatar_prompt(dados[0]))
print(f"Resposta esperada: {dados[0]['output']}")
print()

# ----------------
# 2. DataLoaders |
# ----------------

train_loader, val_loader = criar_loaders_instrucoes(
    dados_treino, dados_val, tokenizer,
    max_length=MAX_LENGTH, batch_size=BATCH_SIZE, device=DEVICE,
)
print(f"Lotes de treino   : {len(train_loader)}")
print(f"Lotes de validação: {len(val_loader)}")

# ----------------------------------
# 3. Carrega o modelo pré-treinado |
# ----------------------------------

print()
print("=" * 60)
print("2. CARREGANDO MODELO PRÉ-TREINADO")
print("=" * 60)

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint não encontrado: {CHECKPOINT}\n"
        "Execute primeiro: python sprints/ch06/experiments/carregar_pesos_pretrained.py"
    )

modelo = GPTModel(CFG_GPT2).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
modelo.load_state_dict(ckpt["model_state_dict"])
modelo.train()

total_params = sum(p.numel() for p in modelo.parameters())
print(f"Parâmetros: {total_params:,}")

# --------------------------------------------
# 4. Geração ANTES do fine-tuning (baseline) |
# --------------------------------------------

print()
print("=" * 60)
print("3. GERAÇÃO ANTES DO FINE-TUNING")
print("=" * 60)

EXEMPLO = dados[0]
prompt_exemplo = formatar_prompt(EXEMPLO)

modelo.eval()
with torch.no_grad():
    ids = text_to_token_ids(prompt_exemplo, tokenizer).to(DEVICE)
    saida = generate(modelo, ids, max_new_tokens=50,
                     context_size=CFG_GPT2["context_length"],
                     temperature=0.8, top_k=25)
texto_antes = token_ids_to_text(saida.cpu(), tokenizer)
resposta_antes = texto_antes[len(prompt_exemplo):]

print(f"Instrução: {EXEMPLO['instruction']}")
print(f"Entrada  : {EXEMPLO.get('input', '')}")
print(f"Esperado : {EXEMPLO['output']}")
print(f"Gerado   : {resposta_antes[:200]}")
modelo.train()

# ----------------
# 5. Fine-tuning |
# ----------------

print()
print("=" * 60)
print("4. FINE-TUNING PARA INSTRUÇÕES")
print("=" * 60)

optimizer = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

historico = treinar_instrucoes(
    model=modelo,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    eval_freq=50,
    tokenizer=tokenizer,
    exemplo_prompt=prompt_exemplo,
)

# ----------------------------------
# 6. Geração DEPOIS do fine-tuning |
# ----------------------------------

print()
print("=" * 60)
print("5. GERAÇÃO DEPOIS DO FINE-TUNING")
print("=" * 60)

modelo.eval()
with torch.no_grad():
    ids = text_to_token_ids(prompt_exemplo, tokenizer).to(DEVICE)
    saida = generate(modelo, ids, max_new_tokens=50,
                     context_size=CFG_GPT2["context_length"],
                     temperature=0.8, top_k=25)
texto_depois = token_ids_to_text(saida.cpu(), tokenizer)
resposta_depois = texto_depois[len(prompt_exemplo):]

print(f"Instrução : {EXEMPLO['instruction']}")
print(f"Entrada   : {EXEMPLO.get('input', '')}")
print(f"Esperado  : {EXEMPLO['output']}")
print(f"Antes FT  : {resposta_antes[:200]}")
print(f"Depois FT : {resposta_depois[:200]}")

# Testa mais 2 exemplos variados
print()
print("Exemplos adicionais:")
for ex in dados[10:12]:
    pr = formatar_prompt(ex)
    with torch.no_grad():
        ids = text_to_token_ids(pr, tokenizer).to(DEVICE)
        saida = generate(modelo, ids, max_new_tokens=60,
                         context_size=CFG_GPT2["context_length"],
                         temperature=0.7, top_k=20)
    resp = token_ids_to_text(saida.cpu(), tokenizer)[len(pr):]
    print(f"\n  Instrução: {ex['instruction']}")
    print(f"  Esperado : {ex['output']}")
    print(f"  Gerado   : {resp[:200]}")

# ---------------------
# 7. Salva resultados |
# ---------------------

resultado = {
    "config": {
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "n_treino": len(dados_treino),
        "n_val": len(dados_val),
    },
    "antes_finetuning": {
        "instrucao": EXEMPLO["instruction"],
        "resposta_esperada": EXEMPLO["output"],
        "resposta_gerada": resposta_antes[:300],
    },
    "depois_finetuning": {
        "instrucao": EXEMPLO["instruction"],
        "resposta_esperada": EXEMPLO["output"],
        "resposta_gerada": resposta_depois[:300],
    },
    "historico_steps": historico["steps"],
    "historico_train_loss": historico["train_losses"],
    "historico_val_loss": historico["val_losses"],
}

RESULTADO.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
print()
print(f"Resultados salvos em: {RESULTADO}")
