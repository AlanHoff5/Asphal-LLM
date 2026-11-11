# Chapter 5 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - laco de treinamento do GPT

# 5.2 (+ apendice D: warmup, cosine decay e gradient clipping)

import math
import time

import torch

from src.model.generation import generate_text_simple, text_to_token_ids, token_ids_to_text
from src.training.losses import calc_loss_batch, calc_loss_loader


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Mede a perda de treino e de validacao usando `eval_iter` lotes de cada.

    model.eval() desliga o dropout para a medida ser justa; no final volta
    para model.train(), porque esta funcao e chamada NO MEIO do treino.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=50):
    """Gera um trecho de texto (decodificacao gulosa) para VER a evolucao do
    modelo ao fim de cada epoca. Devolve o texto gerado."""
    model.eval()
    context_size = model.cfg["context_length"]      # o GPTModel guarda a config
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, max_new_tokens, context_size)
    texto = token_ids_to_text(token_ids.cpu(), tokenizer)
    print("   Amostra:", texto.replace("\n", " "))
    model.train()
    return texto


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """Laco de treino do livro (secao 5.2).

    Para cada epoca e para cada lote:
        1. optimizer.zero_grad()   apaga os gradientes do lote anterior
        2. calc_loss_batch(...)    forward + cross-entropy
        3. loss.backward()         calcula os gradientes (retropropagacao)
        4. optimizer.step()        o AdamW atualiza os pesos
    A cada `eval_freq` passos mede as perdas de treino/validacao e, ao fim de
    cada epoca, gera um texto de exemplo.

    Diferenca em relacao ao livro: alem das tres listas do livro
    (train_losses, val_losses, track_tokens_seen), devolve um dicionario
    `historico` com o passo e a epoca de cada medida, as amostras de texto
    e o tempo de cada epoca, para o relatorio e os graficos.
    """
    historico = {
        "train_losses": [], "val_losses": [], "tokens_seen": [],
        "steps": [], "epocas": [], "amostras": [], "tempo_por_epoca_s": [],
    }
    tokens_seen, global_step = 0, -1
    passos_por_epoca = len(train_loader)

    for epoch in range(num_epochs):
        model.train()                                   # liga o dropout
        inicio_epoca = time.time()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                                           # 1
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 2
            loss.backward()                                                 # 3
            optimizer.step()                                                # 4
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                historico["train_losses"].append(train_loss)
                historico["val_losses"].append(val_loss)
                historico["tokens_seen"].append(tokens_seen)
                historico["steps"].append(global_step)
                historico["epocas"].append((global_step + 1) / passos_por_epoca)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        historico["tempo_por_epoca_s"].append(time.time() - inicio_epoca)
        texto = generate_and_print_sample(model, tokenizer, device, start_context)
        historico["amostras"].append({"epoca": epoch + 1, "texto": texto})

    return historico


def taxa_de_aprendizado(global_step, peak_lr, initial_lr, min_lr, warmup_steps, total_steps):
    """Taxa de aprendizado de cada passo (apendice D do livro).

    - warmup: nos primeiros `warmup_steps` passos sobe em linha reta de
      initial_lr ate peak_lr (passos pequenos enquanto os pesos sao aleatorios);
    - cosine decay: depois desce em meia onda de cosseno ate min_lr
      (passos finos no fim, para nao estragar o que ja foi aprendido).
    """
    if global_step < warmup_steps:
        return initial_lr + global_step * (peak_lr - initial_lr) / warmup_steps
    progresso = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progresso))


def train_model(model, train_loader, val_loader, optimizer, device,
                num_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6, grad_clip=1.0):
    """Laco de treino do apendice D: o mesmo do train_model_simple, mais
    tres tecnicas para deixar o treino mais estavel.

    1. warmup + cosine decay: a taxa de aprendizado muda a cada passo
       (taxa_de_aprendizado). O pico e o lr configurado no otimizador.
    2. gradient clipping: depois do backward, se a norma do gradiente passar
       de `grad_clip`, ele e reduzido ate o limite. Evita os "saltos" da perda
       causados por gradientes enormes. Como no livro, so e aplicado depois
       do warmup.

    Devolve o mesmo `historico` do train_model_simple, mais a lista "lrs"
    com a taxa usada em cada passo.
    """
    historico = {
        "train_losses": [], "val_losses": [], "tokens_seen": [],
        "steps": [], "epocas": [], "amostras": [], "tempo_por_epoca_s": [], "lrs": [],
    }
    tokens_seen, global_step = 0, -1
    passos_por_epoca = len(train_loader)
    total_steps = passos_por_epoca * num_epochs
    peak_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(num_epochs):
        model.train()
        inicio_epoca = time.time()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            lr = taxa_de_aprendizado(global_step, peak_lr, initial_lr, min_lr,
                                     warmup_steps, total_steps)
            for grupo in optimizer.param_groups:
                grupo["lr"] = lr
            historico["lrs"].append(lr)

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            if grad_clip is not None and global_step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                historico["train_losses"].append(train_loss)
                historico["val_losses"].append(val_loss)
                historico["tokens_seen"].append(tokens_seen)
                historico["steps"].append(global_step)
                historico["epocas"].append((global_step + 1) / passos_por_epoca)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, lr {lr:.2e}")

        historico["tempo_por_epoca_s"].append(time.time() - inicio_epoca)
        texto = generate_and_print_sample(model, tokenizer, device, start_context)
        historico["amostras"].append({"epoca": epoch + 1, "texto": texto})

    # restaura o lr de pico no otimizador (o estado salvo fica coerente)
    for grupo in optimizer.param_groups:
        grupo["lr"] = peak_lr
    return historico
