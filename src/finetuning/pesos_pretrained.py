# Chapter 6 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM

# 6.4 (livro) + utilitários do projeto

# Carrega os pesos oficiais do GPT-2 (OpenAI via Hugging Face) para o
# GPTModel do projeto.

# A arquitetura do nosso GPTModel é compatível com o GPT-2 small, mas os
# nomes dos tensores são diferentes. Esta função faz o mapeamento tensor a
# tensor, seguindo a mesma estratégia do livro (seção 5.5 / apêndice B).

# Mapeamento de nomes (HuggingFace → nosso GPTModel):
#   transformer.wte.weight              → embeddings.token_embedding.weight
#   transformer.wpe.weight              → embeddings.positional_embedding.weight
#   transformer.h.N.ln_1.{weight,bias}  → trf_blocks.N.norm1.{scale,shift}
#   transformer.h.N.ln_2.{weight,bias}  → trf_blocks.N.norm2.{scale,shift}
#   transformer.h.N.attn.c_attn.weight  → trf_blocks.N.att.{W_query,W_key,W_value}.weight
#   transformer.h.N.attn.c_attn.bias    → trf_blocks.N.att.{W_query,W_key,W_value}.bias
#   transformer.h.N.attn.c_proj.weight  → trf_blocks.N.att.out_proj.weight
#   transformer.h.N.attn.c_proj.bias    → trf_blocks.N.att.out_proj.bias
#   transformer.h.N.mlp.c_fc.weight     → trf_blocks.N.ff.layers.0.weight
#   transformer.h.N.mlp.c_fc.bias       → trf_blocks.N.ff.layers.0.bias
#   transformer.h.N.mlp.c_proj.weight   → trf_blocks.N.ff.layers.2.weight
#   transformer.h.N.mlp.c_proj.bias     → trf_blocks.N.ff.layers.2.bias
#   transformer.ln_f.weight             → final_norm.scale
#   transformer.ln_f.bias               → final_norm.shift
#   lm_head.weight                      → out_head.weight  (já é weight-tied no HF)

# Observação sobre transposta:
#   Os pesos de Conv1D do GPT-2 original (c_attn, c_proj, c_fc, c_proj da MLP)
#   têm shape (in, out) em vez de (out, in) do nn.Linear do PyTorch. Por isso
#   aplicamos .T (transposta) ao copiá-los.

import torch


# ------------------
# Função principal |
# ------------------

def carregar_pesos_gpt2(gpt_model, gpt2_model_hf):
    # Copia os pesos do modelo HuggingFace para o GPTModel do projeto.

    # Parâmetros
    # ----------
    # gpt_model : GPTModel
    #     Instância do nosso GPTModel com GPT_CONFIG_124M (pesos aleatórios).
    # gpt2_model_hf : transformers.GPT2Model
    #     Modelo carregado com `GPT2Model.from_pretrained("gpt2")`.

    # Retorna
    # -------
    # gpt_model com os pesos do GPT-2 copiados (in-place + return para encadeamento).
    sd_hf  = gpt2_model_hf.state_dict()   # pesos do HuggingFace
    sd_our = gpt_model.state_dict()       # pesos do nosso modelo (aleatórios)

    with torch.no_grad():

        # --- Embeddings -------------------------------------------------------
        sd_our["embeddings.token_embedding.weight"].copy_(
            sd_hf["transformer.wte.weight"]
        )
        sd_our["embeddings.positional_embedding.weight"].copy_(
            sd_hf["transformer.wpe.weight"]
        )

        # --- Transformer Blocks -----------------------------------------------
        for i in range(gpt_model.cfg["n_layers"]):
            hf  = f"transformer.h.{i}"
            our = f"trf_blocks.{i}"

            # LayerNorm 1 (antes da atenção)
            sd_our[f"{our}.norm1.scale"].copy_(sd_hf[f"{hf}.ln_1.weight"])
            sd_our[f"{our}.norm1.shift"].copy_(sd_hf[f"{hf}.ln_1.bias"])

            # LayerNorm 2 (antes da FeedForward)
            sd_our[f"{our}.norm2.scale"].copy_(sd_hf[f"{hf}.ln_2.weight"])
            sd_our[f"{our}.norm2.shift"].copy_(sd_hf[f"{hf}.ln_2.bias"])

            # Atenção: c_attn é a concatenação de Q, K, V (3 * emb_dim)
            # Shape HF: (emb_dim, 3*emb_dim) → transposta → (3*emb_dim, emb_dim)
            c_attn_w = sd_hf[f"{hf}.attn.c_attn.weight"].T  # (3*d, d)
            c_attn_b = sd_hf[f"{hf}.attn.c_attn.bias"]      # (3*d,)
            d = gpt_model.cfg["emb_dim"]

            sd_our[f"{our}.att.W_query.weight"].copy_(c_attn_w[:d, :])
            sd_our[f"{our}.att.W_key.weight"].copy_(c_attn_w[d:2*d, :])
            sd_our[f"{our}.att.W_value.weight"].copy_(c_attn_w[2*d:, :])

            sd_our[f"{our}.att.W_query.bias"].copy_(c_attn_b[:d])
            sd_our[f"{our}.att.W_key.bias"].copy_(c_attn_b[d:2*d])
            sd_our[f"{our}.att.W_value.bias"].copy_(c_attn_b[2*d:])

            # Projeção de saída da atenção (c_proj)
            sd_our[f"{our}.att.out_proj.weight"].copy_(
                sd_hf[f"{hf}.attn.c_proj.weight"].T
            )
            sd_our[f"{our}.att.out_proj.bias"].copy_(
                sd_hf[f"{hf}.attn.c_proj.bias"]
            )

            # FeedForward: c_fc → layers.0, c_proj → layers.2
            sd_our[f"{our}.ff.layers.0.weight"].copy_(
                sd_hf[f"{hf}.mlp.c_fc.weight"].T
            )
            sd_our[f"{our}.ff.layers.0.bias"].copy_(
                sd_hf[f"{hf}.mlp.c_fc.bias"]
            )
            sd_our[f"{our}.ff.layers.2.weight"].copy_(
                sd_hf[f"{hf}.mlp.c_proj.weight"].T
            )
            sd_our[f"{our}.ff.layers.2.bias"].copy_(
                sd_hf[f"{hf}.mlp.c_proj.bias"]
            )

        # --- LayerNorm final --------------------------------------------------
        sd_our["final_norm.scale"].copy_(sd_hf["transformer.ln_f.weight"])
        sd_our["final_norm.shift"].copy_(sd_hf["transformer.ln_f.bias"])

        # --- Cabeça de linguagem (LM head) ------------------------------------
        # No HF o lm_head.weight é o mesmo tensor que wte.weight (weight tying).
        # Copiamos mesmo assim para o out_head do nosso modelo (sem weight tying).
        sd_our["out_head.weight"].copy_(sd_hf["transformer.wte.weight"])

    gpt_model.load_state_dict(sd_our)
    return gpt_model


# ---------------------------------------------------------------
# Utilitário: baixar e retornar o modelo HF pronto para a cópia |
# ---------------------------------------------------------------

def baixar_gpt2_hf(modelo="gpt2"):
    # Baixa o GPT-2 oficial do HuggingFace e devolve o modelo.

    # Requer: pip install transformers
    # Modelos disponíveis: "gpt2" (124M), "gpt2-medium" (355M),
    #                      "gpt2-large" (774M), "gpt2-xl" (1558M).
    
    try:
        from transformers import GPT2Model
    except ImportError:
        raise ImportError(
            "Instale o pacote transformers:\n"
            "    pip install transformers\n"
        )

    print(f"Baixando pesos do '{modelo}' via HuggingFace...")
    hf_model = GPT2Model.from_pretrained(modelo)
    hf_model.eval()
    print("Download concluído.")
    return hf_model


def verificar_mapeamento(gpt_model, gpt2_model_hf, tolerancia=1e-5):
    # Verifica se os pesos foram copiados corretamente comparando
    # alguns tensores selecionados. Imprime OK ou ERRO para cada um.

    # Retorna True se todos passaram, False se algum falhou.

    sd_hf  = gpt2_model_hf.state_dict()
    sd_our = gpt_model.state_dict()
    d = gpt_model.cfg["emb_dim"]

    verificacoes = [
        # (descrição, tensor_nosso, tensor_hf, precisa_transposta)
        ("token embedding",
         sd_our["embeddings.token_embedding.weight"],
         sd_hf["transformer.wte.weight"], False),
        ("positional embedding",
         sd_our["embeddings.positional_embedding.weight"],
         sd_hf["transformer.wpe.weight"], False),
        ("bloco 0 - norm1 scale",
         sd_our["trf_blocks.0.norm1.scale"],
         sd_hf["transformer.h.0.ln_1.weight"], False),
        ("bloco 0 - W_query weight",
         sd_our["trf_blocks.0.att.W_query.weight"],
         sd_hf["transformer.h.0.attn.c_attn.weight"].T[:d, :], False),
        ("bloco 0 - ff layers.0 weight",
         sd_our["trf_blocks.0.ff.layers.0.weight"],
         sd_hf["transformer.h.0.mlp.c_fc.weight"].T, False),
        ("final_norm scale",
         sd_our["final_norm.scale"],
         sd_hf["transformer.ln_f.weight"], False),
    ]

    todos_ok = True
    for descricao, t_nosso, t_hf, transposta in verificacoes:
        t_ref = t_hf.T if transposta else t_hf
        ok = torch.allclose(t_nosso, t_ref, atol=tolerancia)
        status = "✓ OK" if ok else "✗ ERRO"
        if not ok:
            todos_ok = False
        print(f"  {status}  {descricao}")

    return todos_ok
