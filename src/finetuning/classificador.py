# Chapter 6 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM

# 6.5 e 6.6

# Adapta o GPTModel para classificação binária (ou multiclasse).

# Estratégia (seção 6.5 do livro):
#   1. Carrega o GPT pré-treinado.
#   2. Congela TODOS os parâmetros (nenhum gradiente).
#   3. Descongela os últimos `n_blocos_treinaveis` Transformer Blocks
#      e a LayerNorm final.
#   4. Substitui a camada `out_head` por uma `classification_head`
#      com `num_classes` saídas.
#   5. Treina apenas as camadas descongeladas + classification_head.

# Por que usar só o último token?
#   O GPT é causal: o token na posição -1 "viu" todos os tokens anteriores.
#   Sua representação resume o texto inteiro. Pegamos esse vetor e passamos
#   pela classification_head para obter os logits de classe.


import torch
import torch.nn as nn

from src.model.gpt_model import GPTModel


class GPTClassifier(nn.Module):
    # GPT pré-treinado com cabeça de classificação.

    # Parâmetros
    # ----------
    # cfg : dict
    #     Configuração do GPTModel (ex.: GPT_CONFIG_124M com qkv_bias=True).
    # num_classes : int
    #     Número de classes (2 para spam/ham, mais para multiclasse).
    # n_blocos_treinaveis : int
    #     Quantos Transformer Blocks finais ficam treináveis.
    #     0 → só a classification_head é treinada.
    #     -1 → todos os blocos são treináveis (fine-tuning completo).

    def __init__(self, cfg, num_classes=2, n_blocos_treinaveis=1):
        super().__init__()
        self.gpt = GPTModel(cfg)
        emb_dim = cfg["emb_dim"]

        # Substitui lm_head por uma cabeça de classificação
        self.classification_head = nn.Linear(emb_dim, num_classes)

        # Congela tudo primeiro
        for param in self.gpt.parameters():
            param.requires_grad = False

        # Descongela os últimos blocos e a LayerNorm final
        if n_blocos_treinaveis > 0:
            n = cfg["n_layers"]
            for bloco in self.gpt.trf_blocks[n - n_blocos_treinaveis:]:
                for param in bloco.parameters():
                    param.requires_grad = True

        for param in self.gpt.final_norm.parameters():
            param.requires_grad = True

        # classification_head sempre treinável
        for param in self.classification_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids):
        # input_ids: (batch, num_tokens)
        x = self.gpt(input_ids)             # (batch, num_tokens, emb_dim)
        # Pega a representação do ÚLTIMO token
        ultimo = x[:, -1, :]                # (batch, emb_dim)
        logits = self.classification_head(ultimo)  # (batch, num_classes)
        return logits

    def contar_parametros_treinaveis(self):
        # Retorna (treináveis, total).
        treinaveis = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total      = sum(p.numel() for p in self.parameters())
        return treinaveis, total


# ------------------------------------------
# Utilitários de treino para classificação |
# ------------------------------------------

def calcular_acuracia_lote(logits, rotulos):
    # Acurácia de um único lote.
    predicoes = torch.argmax(logits, dim=-1)
    return (predicoes == rotulos).float().mean().item()


def avaliar_classificador(model, loader, device):
    # Acurácia média sobre todos os lotes de um DataLoader.

    # Chama model.eval() internamente; não muda o modo de treino.
    model.eval()
    total_corretos = 0
    total_amostras = 0
    with torch.no_grad():
        for input_ids, rotulos in loader:
            input_ids = input_ids.to(device)
            rotulos   = rotulos.to(device)
            logits    = model(input_ids)
            predicoes = torch.argmax(logits, dim=-1)
            total_corretos += (predicoes == rotulos).sum().item()
            total_amostras += rotulos.size(0)
    model.train()
    return total_corretos / total_amostras if total_amostras > 0 else 0.0


def treinar_classificador(model, train_loader, val_loader, optimizer,
                           device, num_epochs, eval_freq=50):
    # Laço de treino para classificação.

    # Semelhante ao train_model_simple do Cap. 5, mas:
    # - usa cross-entropy sobre `num_classes` classes;
    # - mede acurácia (não perda de linguagem);
    #- não gera texto de exemplo.

    # Retorna histórico: {train_losses, val_losses, train_accs, val_accs, steps}.
    historico = {
        "train_losses": [], "val_losses": [],
        "train_accs":   [], "val_accs":   [],
        "steps": [],
    }
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        for input_ids, rotulos in train_loader:
            input_ids = input_ids.to(device)
            rotulos   = rotulos.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss   = nn.functional.cross_entropy(logits, rotulos)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_freq == 0:
                # Avalia treino (só 1 lote para ser rápido) e validação
                model.eval()
                with torch.no_grad():
                    train_loss = loss.item()
                    train_acc  = calcular_acuracia_lote(logits, rotulos)
                val_loss = _loss_loader(model, val_loader, device)
                val_acc  = avaliar_classificador(model, val_loader, device)
                model.train()

                historico["train_losses"].append(train_loss)
                historico["val_losses"].append(val_loss)
                historico["train_accs"].append(train_acc)
                historico["val_accs"].append(val_acc)
                historico["steps"].append(global_step)

                print(f"Ep {epoch+1} Step {global_step:05d} | "
                      f"Loss treino {train_loss:.3f} val {val_loss:.3f} | "
                      f"Acc treino {train_acc:.3f} val {val_acc:.3f}")

    return historico


def _loss_loader(model, loader, device, max_batches=None):
    # Perda média de cross-entropy num DataLoader de classificação.
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, (input_ids, rotulos) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            logits = model(input_ids.to(device))
            loss   = nn.functional.cross_entropy(logits, rotulos.to(device))
            total += loss.item()
            n     += 1
    model.train()
    return total / n if n > 0 else float("nan")
