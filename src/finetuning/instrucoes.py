# Chapter 7 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM 

# 7.2, 7.3 e 7.4

# Prepara dados e laço de treino para instruction fine-tuning no formato Alpaca.

"""
Formato de cada exemplo no dataset JSON:
    {
        "instruction": "Rewrite the following sentence ...",
        "input": "The quick brown fox ...",       ← pode ser vazio ""
        "output": "A swift auburn fox ..."
    }

O prompt formatado varia conforme `input`:
  Com input:
      Below is an instruction that describes a task, paired with an input
      that provides further context. Write a response that appropriately
      completes the request.

      ### Instruction:
      {instruction}

      ### Input:
      {input}

      ### Response:

  Sem input:
      Below is an instruction that describes a task. Write a response
      that appropriately completes the request.

      ### Instruction:
      {instruction}

      ### Response:
"""

# O modelo aprende a prever APENAS os tokens da resposta (os tokens do prompt
# recebem target=-100 para serem ignorados pela cross-entropy).


import json
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------
# URL do dataset de instruções (mesmo do livro — 1.100 exemplos) |
# ----------------------------------------------------------------

URL_INSTRUCOES = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)


def baixar_dataset_instrucoes(diretorio: Path) -> Path:
    # Baixa o JSON de instruções do repositório do livro.
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)
    destino = diretorio / "instruction-data.json"

    if destino.exists():
        print(f"Dataset já existe: {destino}")
        return destino

    print("Baixando dataset de instruções...")
    urllib.request.urlretrieve(URL_INSTRUCOES, destino)
    print(f"Salvo em: {destino}")
    return destino


def carregar_instrucoes(caminho: Path):
    # Carrega a lista de dicts {instruction, input, output}.
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------
# Formatação do prompt (estilo Alpaca) |
# --------------------------------------

def formatar_prompt(exemplo: dict) -> str:
    # Monta o texto completo do prompt para um exemplo.

    # Se `input` não for vazio, usa o template com contexto;
    # caso contrário usa o template sem contexto.
    
    instrucao = exemplo["instruction"]
    entrada   = exemplo.get("input", "").strip()

    if entrada:
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instrucao}\n\n"
            f"### Input:\n{entrada}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instrucao}\n\n"
        "### Response:\n"
    )


def formatar_exemplo_completo(exemplo: dict) -> str:
    # Prompt + resposta (usado para tokenizar o exemplo inteiro).
    return formatar_prompt(exemplo) + exemplo["output"]


# -----------------
# Dataset PyTorch |
# -----------------

class InstructionDataset(Dataset):
    # Dataset tokenizado para instruction fine-tuning.

    # Cada item é (input_ids, target_ids) onde:
    # - input_ids  : tokens do prompt + resposta (sem o último token)
    # - target_ids : tokens do prompt + resposta deslocados (sem o primeiro token);
    #                   as posições do prompt recebem -100 para serem ignoradas.

    # Assim o modelo aprende a prever APENAS os tokens da resposta,
    # não os tokens do prompt que ele já conhece.

    IGNORE_INDEX = -100   # valor ignorado pelo F.cross_entropy

    def __init__(self, dados, tokenizer, max_length=1024):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.amostras   = []

        for ex in dados:
            prompt_completo = formatar_exemplo_completo(ex)
            so_prompt       = formatar_prompt(ex)

            ids_completo = tokenizer.encode(prompt_completo)
            ids_prompt   = tokenizer.encode(so_prompt)

            # Trunca se necessário
            ids_completo = ids_completo[:max_length + 1]

            input_ids  = ids_completo[:-1]   # todos menos o último
            target_ids = ids_completo[1:]    # todos menos o primeiro

            # Mascara os tokens do prompt nos targets
            n_prompt = min(len(ids_prompt), len(target_ids))
            target_ids_mascarados = (
                [self.IGNORE_INDEX] * n_prompt
                + target_ids[n_prompt:]
            )

            self.amostras.append((
                torch.tensor(input_ids,            dtype=torch.long),
                torch.tensor(target_ids_mascarados, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, idx):
        return self.amostras[idx]


# ----------------------------------------------
# Collate function (padding dinâmico por lote) |
# ----------------------------------------------

def collate_instrucoes(lote, pad_id=50256, device="cpu"):
    # Padeia as sequências do lote até o comprimento máximo do lote.

    # - input_ids  recebem pad_id nas posições extras.
    # - target_ids recebem IGNORE_INDEX (-100) nas posições extras,
    #   para que o padding não contribua para a perda.
    inputs, targets = zip(*lote)
    max_len = max(t.size(0) for t in inputs)

    inputs_pad  = []
    targets_pad = []

    for inp, tgt in zip(inputs, targets):
        pad = max_len - inp.size(0)
        inputs_pad.append(
            torch.cat([inp, torch.full((pad,), pad_id, dtype=torch.long)])
        )
        targets_pad.append(
            torch.cat([tgt, torch.full((pad,), -100, dtype=torch.long)])
        )

    return (
        torch.stack(inputs_pad).to(device),
        torch.stack(targets_pad).to(device),
    )


def criar_loaders_instrucoes(dados_treino, dados_val, tokenizer,
                              max_length=512, batch_size=4,
                              num_workers=0, device="cpu"):
    # Cria DataLoaders de treino e validação com collate de padding dinâmico.
    ds_treino = InstructionDataset(dados_treino, tokenizer, max_length)
    ds_val    = InstructionDataset(dados_val,    tokenizer, max_length)

    fn = lambda b: collate_instrucoes(b, device=device)

    return (
        DataLoader(ds_treino, batch_size=batch_size, shuffle=True,
                   collate_fn=fn, num_workers=num_workers),
        DataLoader(ds_val,    batch_size=batch_size, shuffle=False,
                   collate_fn=fn, num_workers=num_workers),
    )


# --------------------------------
# Laço de treino para instruções |
# --------------------------------

def treinar_instrucoes(model, train_loader, val_loader, optimizer,
                        device, num_epochs, eval_freq=50, tokenizer=None,
                        exemplo_prompt=None):
    # Laço de treino idêntico ao train_model_simple (Cap. 5), mas:

    # - targets com -100 são ignorados pelo cross_entropy automaticamente
    #   (PyTorch já faz isso quando ignore_index=-100).
    # - Opcionalmente gera uma resposta de exemplo a cada época.
    from src.model.generation import generate, text_to_token_ids, token_ids_to_text

    historico = {
        "train_losses": [], "val_losses": [], "steps": [],
    }
    global_step = 0

    for epoch in range(num_epochs):
        model.train()

        for input_ids, target_ids in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids)         # (b, n, vocab)
            # Achata e ignora -100 automaticamente
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_ids.flatten(),
                ignore_index=-100,
            )
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_freq == 0:
                val_loss = _loss_instrucoes(model, val_loader)
                historico["train_losses"].append(loss.item())
                historico["val_losses"].append(val_loss)
                historico["steps"].append(global_step)
                print(f"Ep {epoch+1} Step {global_step:05d} | "
                      f"Loss treino {loss.item():.3f} val {val_loss:.3f}")

        # Gera resposta de exemplo ao fim de cada época
        if tokenizer is not None and exemplo_prompt is not None:
            model.eval()
            with torch.no_grad():
                ids = text_to_token_ids(exemplo_prompt, tokenizer).to(device)
                ctx = model.cfg["context_length"] if hasattr(model, "cfg") else 1024
                saida = generate(model, ids, max_new_tokens=50,
                                 context_size=ctx, temperature=0.8, top_k=25)
            resp = token_ids_to_text(saida.cpu(), tokenizer)
            resp_so = resp[len(exemplo_prompt):]
            print(f"   Exemplo época {epoch+1}: ...{resp_so[:120]}")
            model.train()

    return historico


def _loss_instrucoes(model, loader, max_batches=5):
    # Perda de validação (rápida — usa até max_batches lotes).
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, (inp, tgt) in enumerate(loader):
            if i >= max_batches:
                break
            logits = model(inp)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), tgt.flatten(), ignore_index=-100
            )
            total += loss.item()
            n += 1
    model.train()
    return total / n if n > 0 else float("nan")
