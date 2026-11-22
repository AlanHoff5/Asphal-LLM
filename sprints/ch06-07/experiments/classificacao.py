# Chapter 6 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM 

# Fine-tuning do GPT-2 pré-treinado para classificar mensagens spam/ham.

# Requer que `carregar_pesos_pretrained.py` já tenha rodado e gerado
#   data/gpt2_pretrained.pth

# Uso:
#     python sprints/ch06/experiments/classificacao.py

# Saída:
#     Acurácia e F1-score no conjunto de teste
#     sprints/ch06/experiments/resultados/classificacao_resultados.json

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import tiktoken

from src.model.config import GPT_CONFIG_124M
from src.training.dispositivo import get_device
from src.finetuning.classificador import (
    GPTClassifier,
    treinar_classificador,
    avaliar_classificador,
)
from src.finetuning.dados_cls import (
    baixar_dataset_spam,
    carregar_spam_df,
    balancear_classes,
    dividir_treino_val_teste,
    criar_loaders_spam,
)

# --------------
# Configuração |
# --------------

DEVICE     = get_device()
DATA_DIR   = PROJECT_ROOT / "data"
CHECKPOINT = DATA_DIR / "gpt2_pretrained.pth"
RESULTADO  = Path(__file__).parent / "resultados" / "classificacao_resultados.json"
RESULTADO.parent.mkdir(parents=True, exist_ok=True)

CFG_GPT2 = dict(GPT_CONFIG_124M)
CFG_GPT2["qkv_bias"] = True

BATCH_SIZE          = 8
NUM_EPOCHS          = 5
LEARNING_RATE       = 5e-5
N_BLOCOS_TREINAVEIS = 1          # congela tudo exceto último bloco + classification_head
MAX_LENGTH          = 128        # trunca mensagens longas (economiza VRAM)

# ------------
# 1. Dataset |
# ------------

print("=" * 60)
print("1. PREPARANDO O DATASET")
print("=" * 60)

tokenizer   = tiktoken.get_encoding("gpt2")
arquivo_tsv = baixar_dataset_spam(DATA_DIR)
dados       = carregar_spam_df(arquivo_tsv)
dados       = balancear_classes(dados)
treino, val, teste = dividir_treino_val_teste(dados)

print(f"Total (balanceado): {len(dados)}  |  spam: {sum(r for r,_ in dados)}  "
      f"ham: {sum(1-r for r,_ in dados)}")
print(f"Treino: {len(treino)}  Validação: {len(val)}  Teste: {len(teste)}")

train_loader, val_loader, test_loader = criar_loaders_spam(
    treino, val, teste, tokenizer, max_length=MAX_LENGTH, batch_size=BATCH_SIZE
)

# -----------------------------------------------------------------------
# 2. Modelo: carrega pesos pré-treinados e adiciona classification_head |
# -----------------------------------------------------------------------

print()
print("=" * 60)
print("2. CONSTRUINDO O CLASSIFICADOR")
print("=" * 60)

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint não encontrado: {CHECKPOINT}\n"
        "Execute primeiro: python sprints/ch06/experiments/carregar_pesos_pretrained.py"
    )

modelo = GPTClassifier(CFG_GPT2, num_classes=2,
                       n_blocos_treinaveis=N_BLOCOS_TREINAVEIS).to(DEVICE)

# Carrega os pesos pré-treinados no GPT interno
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
modelo.gpt.load_state_dict(ckpt["model_state_dict"])

treinaveis, total = modelo.contar_parametros_treinaveis()
print(f"Parâmetros treináveis : {treinaveis:,}  ({100*treinaveis/total:.1f}% do total)")
print(f"Parâmetros congelados : {total - treinaveis:,}")

# ---------------------------------------------
# 3. Acurácia ANTES do fine-tuning (baseline) |
# ---------------------------------------------

print()
print("=" * 60)
print("3. BASELINE (antes do fine-tuning)")
print("=" * 60)

acc_treino_antes = avaliar_classificador(modelo, train_loader, DEVICE)
acc_val_antes    = avaliar_classificador(modelo, val_loader,   DEVICE)
acc_teste_antes  = avaliar_classificador(modelo, test_loader,  DEVICE)
print(f"Acurácia treino   : {acc_treino_antes:.3f}")
print(f"Acurácia validação: {acc_val_antes:.3f}")
print(f"Acurácia teste    : {acc_teste_antes:.3f}")
print("(≈ 0.50 é o esperado — a classification_head ainda é aleatória)")

# ----------------
# 4. Fine-tuning |
# ----------------

print()
print("=" * 60)
print("4. FINE-TUNING")
print("=" * 60)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, modelo.parameters()),
    lr=LEARNING_RATE, weight_decay=0.1
)

historico = treinar_classificador(
    modelo, train_loader, val_loader, optimizer, DEVICE,
    num_epochs=NUM_EPOCHS, eval_freq=50
)

# ---------------------------------------------------
# 5. Avaliação DEPOIS do fine-tuning (Entregável 4) |
# ---------------------------------------------------

print()
print("=" * 60)
print("5. AVALIAÇÃO FINAL (Entregável 4)")
print("=" * 60)

acc_treino = avaliar_classificador(modelo, train_loader, DEVICE)
acc_val    = avaliar_classificador(modelo, val_loader,   DEVICE)
acc_teste  = avaliar_classificador(modelo, test_loader,  DEVICE)

# F1-score, precisão e revocação (manual para não depender de sklearn)
def metricas_binarias(model, loader, device):
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for ids, labels in loader:
            preds = torch.argmax(model(ids.to(device)), dim=-1).cpu()
            for p, l in zip(preds, labels):
                if p == 1 and l == 1: tp += 1
                elif p == 1 and l == 0: fp += 1
                elif p == 0 and l == 1: fn += 1
                else: tn += 1
    model.train()
    precisao  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    revocacao = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precisao * revocacao / (precisao + revocacao)
          if (precisao + revocacao) > 0 else 0.0)
    return {"precisao": precisao, "revocacao": revocacao, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

metr = metricas_binarias(modelo, test_loader, DEVICE)

print(f"Acurácia  treino   : {acc_treino_antes:.3f} → {acc_treino:.3f}")
print(f"Acurácia  validação: {acc_val_antes:.3f}    → {acc_val:.3f}")
print(f"Acurácia  teste    : {acc_teste_antes:.3f}  → {acc_teste:.3f}")
print()
print(f"Precisão  (teste)  : {metr['precisao']:.3f}")
print(f"Revocação (teste)  : {metr['revocacao']:.3f}")
print(f"F1-score  (teste)  : {metr['f1']:.3f}")
print()
print(f"Matriz de confusão (teste):")
print(f"  VP={metr['tp']}  FP={metr['fp']}")
print(f"  FN={metr['fn']}  VN={metr['tn']}")

# ---------------------
# 6. Salva resultados |
# ---------------------

resultado = {
    "config": {
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "n_blocos_treinaveis": N_BLOCOS_TREINAVEIS,
        "max_length": MAX_LENGTH,
    },
    "antes_finetuning": {
        "acc_treino": acc_treino_antes,
        "acc_val":    acc_val_antes,
        "acc_teste":  acc_teste_antes,
    },
    "depois_finetuning": {
        "acc_treino": acc_treino,
        "acc_val":    acc_val,
        "acc_teste":  acc_teste,
        **metr,
    },
    "historico_steps": historico["steps"],
    "historico_val_acc": historico["val_accs"],
}

RESULTADO.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
print()
print(f"Resultados salvos em: {RESULTADO}")
