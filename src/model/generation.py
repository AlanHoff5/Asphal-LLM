# Chapter 4 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - inferencia e geracao autoregressiva de texto

# 4.7 (+ temperatura e top-k, que o livro apresenta na secao 5.3)

import torch


def text_to_token_ids(texto, tokenizer):
    # Texto -> tensor (1, num_tokens). unsqueeze(0) cria a dimensao de lote.
    ids = tokenizer.encode(texto, allowed_special={"<|endoftext|>"})
    return torch.tensor(ids).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    # Tensor (1, num_tokens) -> texto. squeeze(0) remove a dimensao de lote.
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Geracao autoregressiva com decodificacao GULOSA.

    idx: (batch, num_tokens) com os Token IDs do contexto inicial.

    A cada iteracao:
      1. corta o contexto para os ultimos `context_size` tokens (o modelo so
         tem Positional Embedding para essas posicoes);
      2. roda o modelo SEM gradientes (so inferencia) -> logits (b, n, vocab);
      3. pega so a ULTIMA posicao -> (b, vocab). Pela mascara causal, e a
         unica que "viu" o texto inteiro, e ela preve o proximo token;
      4. softmax -> probabilidades (positivas, somam 1);
      5. argmax -> o token mais provavel, shape (b, 1);
      6. concatena ao contexto e repete.

    O softmax e opcional aqui (nao muda a ordem dos valores), mas fica para
    deixar explicita a passagem logits -> probabilidades.

    chamar model.eval() antes, para desligar o dropout.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]                      # 1
        with torch.no_grad():
            logits = model(idx_cond)                           # 2  (b, n, vocab)
        logits = logits[:, -1, :]                              # 3  (b, vocab)
        probas = torch.softmax(logits, dim=-1)                 # 4  (b, vocab)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # 5  (b, 1)
        idx = torch.cat((idx, idx_next), dim=1)                # 6  (b, n + 1)
    return idx


def aplicar_top_k(logits, top_k):
    """Mantem so os `top_k` maiores logits de cada linha; os demais viram
    -inf, que o softmax transforma em probabilidade 0."""
    valores_top, _ = torch.topk(logits, top_k, dim=-1)
    limite = valores_top[:, -1:]                               # k-esimo maior, (b, 1)
    return torch.where(logits < limite, torch.full_like(logits, -torch.inf), logits)


def probabilidades_do_proximo_token(logits, temperature=1.0, top_k=None):
    """Transforma os logits da ultima posicao (b, vocab) nas probabilidades
    usadas para sortear o proximo token: top-k -> /temperatura -> softmax.

    temperature < 1 concentra a probabilidade no favorito (mais previsivel);
    temperature > 1 achata a distribuicao (mais variado, mais absurdos).
    """
    if top_k is not None:
        logits = aplicar_top_k(logits, top_k)
    return torch.softmax(logits / temperature, dim=-1)


def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """Geracao autoregressiva com estrategias de selecao do proximo token.

    - temperature == 0  -> decodificacao gulosa (argmax), igual a
      generate_text_simple (apos aplicar o top-k, se houver);
    - temperature > 0   -> SORTEIO com torch.multinomial a partir de
      softmax(logits / temperature);
    - top_k             -> so os k tokens mais provaveis podem ser sorteados
      (top_k=1 equivale a gulosa);
    - eos_id            -> para antes se o token de fim de texto for gerado
      (so faz sentido com batch=1).

    Para resultados reproduziveis com sorteio, use torch.manual_seed antes.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature > 0.0:
            probas = probabilidades_do_proximo_token(logits, temperature, top_k)
            idx_next = torch.multinomial(probas, num_samples=1)   # (b, 1)
        else:
            if top_k is not None:
                logits = aplicar_top_k(logits, top_k)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx.shape[0] == 1 and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx
