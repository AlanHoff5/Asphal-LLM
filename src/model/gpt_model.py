# Chapter 4 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - modelo GPT completo

# 4.6

from torch import nn

from src.data.embeddings import GPTInputEmbeddings
from src.model.layer_norm import LayerNorm
from src.model.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """Arquitetura GPT (decoder-only) completa.

        Token IDs                               (batch, num_tokens)
          -> Token + Positional Embedding       (batch, num_tokens, emb_dim)   [Sprint 2]
          -> Dropout
          -> n_layers x TransformerBlock        (batch, num_tokens, emb_dim)   [Sprint 3 + 4]
          -> LayerNorm final                    (batch, num_tokens, emb_dim)
          -> Camada de saida (LM head)          (batch, num_tokens, vocab_size) = LOGITS

    Diferenca em relacao ao livro: em vez de criar `tok_emb` e `pos_emb`
    aqui, reaproveitamos o `GPTInputEmbeddings` da Sprint 2, que faz
    exatamente a soma token + posicao (e ja valida o tamanho do contexto).
    O numero de parametros e o mesmo do livro.

    - `trf_blocks`: nn.Sequential com n_layers blocos DIFERENTES (cada um
      com seus proprios pesos).
    - `final_norm`: necessaria porque, no Pre-LayerNorm, a saida do ultimo
      bloco ainda nao foi normalizada.
    - `out_head`: Linear(emb_dim -> vocab_size) sem bias. Para cada posicao
      t, produz uma pontuacao (logit) para cada token do vocabulario ser o
      token da posicao t + 1.
    - Weight tying (opcional, cfg["weight_tying"]=True): a camada de saida
      usa a MESMA matriz do token embedding (ambas tem shape
      (vocab_size, emb_dim)), como no GPT-2 original. O livro deixa
      desligado no Cap. 4 (163M parametros); ligado da os 124M do nome.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)

        self.embeddings = GPTInputEmbeddings(
            vocab_size=cfg["vocab_size"],
            context_length=cfg["context_length"],
            embedding_dim=cfg["emb_dim"],
        )
        self.drop_emb = nn.Dropout(cfg.get("drop_rate_emb", cfg["drop_rate"]))

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        if cfg.get("weight_tying", False):
            self.out_head.weight = self.embeddings.token_embedding.weight

    def forward(self, in_idx):
        # in_idx: (batch, num_tokens) com Token IDs
        x = self.embeddings(in_idx)      # (batch, num_tokens, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)           # (batch, num_tokens, emb_dim)
        x = self.final_norm(x)           # (batch, num_tokens, emb_dim)
        logits = self.out_head(x)        # (batch, num_tokens, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# Utilitarios de contagem de parametros e memoria (secao 4.6 e Exercicio 4.2)
# ---------------------------------------------------------------------------

def contar_parametros(model):
    """Total de parametros. `model.parameters()` nao repete tensores
    compartilhados, entao com weight tying a matriz conta uma vez so."""
    return sum(p.numel() for p in model.parameters())


def contar_parametros_sem_saida(model):
    """Total descontando a camada de saida - e assim que o livro chega ao
    "124M" sem implementar weight tying."""
    total = contar_parametros(model)
    if model.out_head.weight is model.embeddings.token_embedding.weight:
        return total
    return total - model.out_head.weight.numel()


def tamanho_em_mb(model, bytes_por_parametro=4):
    """Memoria so dos pesos (float32 = 4 bytes). Treinar exige bem mais:
    gradientes, estado do otimizador e ativacoes."""
    return contar_parametros(model) * bytes_por_parametro / (1024 * 1024)


def parametros_esperados(cfg):
    """Conta os parametros a partir da configuracao, sem instanciar o
    modelo. Util para modelos grandes demais para a memoria (ex.: GPT-2 XL)
    e para conferir a implementacao nos testes.

    Devolve um dicionario com o total e a divisao por componente.
    """
    d = cfg["emb_dim"]
    v = cfg["vocab_size"]
    c = cfg["context_length"]
    h = cfg.get("ff_hidden_dim", 4 * d)

    qkv = 3 * d * d + (3 * d if cfg["qkv_bias"] else 0)
    atencao = qkv + d * d + d                     # Q, K, V + out_proj (com bias)
    feed_forward = d * h + h + h * d + d          # duas Linear com bias
    layer_norms_bloco = 2 * (2 * d)               # norm1 + norm2 (scale + shift)
    por_bloco = atencao + feed_forward + layer_norms_bloco

    partes = {
        "token_embedding": v * d,
        "positional_embedding": c * d,
        "transformer_blocks": cfg["n_layers"] * por_bloco,
        "final_norm": 2 * d,
        "out_head": 0 if cfg.get("weight_tying", False) else d * v,
    }
    partes["total"] = sum(partes.values())
    partes["por_bloco"] = por_bloco
    partes["atencao_por_bloco"] = atencao
    partes["feed_forward_por_bloco"] = feed_forward
    return partes
