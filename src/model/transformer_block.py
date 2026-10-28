# Chapter 4 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - Transformer Block

# 4.4 e 4.5

from torch import nn

from src.attention.multi_head_attention import MultiHeadAttention
from src.model.feed_forward import FeedForward
from src.model.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """Unidade que se repete dentro do GPT (12x no GPT-2 small).

    Junta as pecas das Sprints anteriores e deste capitulo:
      - Multi-Head Attention da Sprint 3 (src/attention/), que ja traz a
        mascara causal;
      - FeedForward com GELU (secao 4.3);
      - duas LayerNorm (secao 4.2);
      - dropout antes de cada soma;
      - duas conexoes de atalho / residuais (secao 4.4).

    Estrutura (Pre-LayerNorm, como no GPT-2):

        x ──┬─> LayerNorm 1 -> Multi-Head Attention -> Dropout ──(+)──> x'
            └──────────────────── atalho ──────────────────────────┘
        x'──┬─> LayerNorm 2 -> FeedForward (GELU)   -> Dropout ──(+)──> saida
            └──────────────────── atalho ──────────────────────────┘

    Padrao de cada metade: NORMALIZA -> PROCESSA -> DROPOUT -> + ATALHO.

    Por que o atalho (x + camada(x))? Cria um caminho curto para o gradiente
    voltar ate as primeiras camadas no backpropagation, evitando o
    "vanishing gradient". Para a soma funcionar, a saida da atencao e da
    FeedForward precisa ter o MESMO shape da entrada: por isso d_in = d_out
    = emb_dim na atencao, e a FeedForward volta para emb_dim.

    Shapes: entrada (batch, num_tokens, emb_dim) -> saida (batch, num_tokens, emb_dim).
    Como o shape nao muda, os blocos podem ser empilhados.
    """

    def __init__(self, cfg):
        super().__init__()
        drop_attn = cfg.get("drop_rate_attn", cfg["drop_rate"])
        drop_shortcut = cfg.get("drop_rate_shortcut", cfg["drop_rate"])

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=drop_attn,
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(drop_shortcut)

    def forward(self, x):
        # --- Parte 1: atencao (mistura informacao entre os tokens) ---
        atalho = x                       # (b, n, emb)
        x = self.norm1(x)                # (b, n, emb)  media 0 / var 1 por token
        x = self.att(x)                  # (b, n, emb)  vetores de contexto
        x = self.drop_shortcut(x)
        x = x + atalho                   # (b, n, emb)  soma a entrada de volta

        # --- Parte 2: FeedForward (processa cada token separadamente) ---
        atalho = x
        x = self.norm2(x)                # (b, n, emb)
        x = self.ff(x)                   # (b, n, emb)  passando por (b, n, 4*emb)
        x = self.drop_shortcut(x)
        x = x + atalho                   # (b, n, emb)
        return x
