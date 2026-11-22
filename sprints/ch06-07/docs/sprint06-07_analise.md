# Sprint 6 — Fine-Tuning para Tarefas Específicas

## 1. Objetivo

A Sprint 5 terminou com um GPT treinado do zero em *The Verdict* — um corpus
pequeno, com overfitting evidente após a época 5. A Sprint 6 muda a
estratégia: em vez de treinar do zero, **partimos dos pesos oficiais do GPT-2**
(treinado pela OpenAI em 1,5 bilhão de tokens da web) e adaptamos o modelo
para duas tarefas específicas:

1. **Classificação** de mensagens spam/ham (Capítulo 6)
2. **Seguir instruções** em linguagem natural (Capítulo 7)

Essa abordagem — carregar pesos pré-treinados e ajustá-los numa tarefa menor —
chama-se **Transfer Learning** e é a base de quase todos os LLMs modernos.

```text
GPT-2 (pesos OpenAI, 124M parâmetros)
  ↓
Carregamento tensor a tensor → GPTModel do projeto
  ↓
Fine-tuning para classificação          Fine-tuning para instruções
  congelar blocos 0–10                    todos os parâmetros treináveis
  treinar bloco 11 + classification_head  cross-entropy c/ ignore_index=-100
  cross-entropy binária                   dataset Alpaca (instrução/entrada/saída)
  ↓                                       ↓
Acurácia, F1, matriz de confusão        Comparação resposta antes × depois
```

O código reutilizável fica em `src/finetuning/` (cumulativo, como os módulos anteriores):

| Arquivo | Conteúdo |
|---|---|
| `pesos_pretrained.py` | mapeamento de tensores HuggingFace → GPTModel, verificação |
| `classificador.py` | `GPTClassifier`, congelamento, laço de treino, métricas |
| `dados_cls.py` | SMS Spam Collection, balanceamento, `SpamDataset` com padding |
| `instrucoes.py` | formato Alpaca, mascaramento do prompt, padding dinâmico, laço |

## 2. Entregável 2 — Carregamento de pesos pré-treinados

### 2.1 O problema do mapeamento

O GPT-2 oficial (HuggingFace) usa nomes de tensor diferentes dos nossos:

| HuggingFace | Nosso GPTModel |
|---|---|
| `transformer.wte.weight` | `embeddings.token_embedding.weight` |
| `transformer.wpe.weight` | `embeddings.position_embedding.weight` |
| `transformer.h.N.ln_1.weight` | `trf_blocks.N.norm1.scale` |
| `transformer.h.N.attn.c_attn.weight` | `trf_blocks.N.att.W_query/W_key/W_value.weight` |
| `transformer.h.N.mlp.c_fc.weight` | `trf_blocks.N.ff.layers.0.weight` |
| `transformer.ln_f.weight` | `final_norm.scale` |

### 2.2 Transposta dos pesos de atenção

Os pesos `c_attn`, `c_proj`, `c_fc` e `c_proj` da MLP do GPT-2 original
são **Conv1D** com shape `(in, out)`, enquanto o `nn.Linear` do PyTorch
espera `(out, in)`. Por isso aplicamos `.T` ao copiá-los — sem isso os
shapes bateriam mas os valores estariam errados.

### 2.3 Divisão do c_attn

O `c_attn` do HuggingFace concatena Query, Key e Value num único tensor
`(emb_dim, 3 * emb_dim)`. Após a transposta fica `(3*d, d)` e dividimos:

```python
c_attn_w = sd_hf["transformer.h.N.attn.c_attn.weight"].T  # (3*d, d)
W_query = c_attn_w[:d, :]
W_key   = c_attn_w[d:2*d, :]
W_value = c_attn_w[2*d:, :]
```

### 2.4 Diferença entre pesos aleatórios e pré-treinados

| | Pesos aleatórios | Pesos pré-treinados |
|---|---|---|
| Geração para "Every effort moves you" | texto sem sentido (tokens aleatórios) | texto em inglês fluente |
| Perda de linguagem | ≈ ln(50257) ≈ 10,82 | ≈ 3–4 (já conhece a língua) |
| Útil para fine-tuning? | não (precisa treinar do zero) | sim (já sabe inglês) |

## 3. Entregável 3 — Fine-tuning para classificação

### 3.1 Adaptação da arquitetura

O `GPTClassifier` substitui a `out_head` (que projeta para 50.257 tokens)
por uma `classification_head` linear com 2 saídas (spam/ham). A previsão
usa **apenas o último token** da sequência — ele "viu" todos os anteriores
e sua representação resume o texto inteiro.

```
GPT(input_ids) → (batch, num_tokens, 768)
                → [:, -1, :]              ← último token
                → (batch, 768)
classification_head → (batch, 2)          ← logits de classe
```

### 3.2 Congelamento de camadas

Com `n_blocos_treinaveis=1` apenas **~0,8 %** dos parâmetros são treináveis:

| Componente | Treináveis? |
|---|---|
| Embeddings | ❌ congelado |
| Transformer Blocks 0–10 | ❌ congelados |
| Transformer Block 11 (último) | ✅ treinável |
| `final_norm` | ✅ treinável |
| `classification_head` | ✅ sempre treinável |

Isso evita **catastrophic forgetting**: o modelo preserva o que aprendeu no
pré-treino e adapta apenas as camadas necessárias para a nova tarefa.

### 3.3 Dataset SMS Spam Collection

- **Fonte**: UCI Machine Learning Repository (5.572 mensagens)
- **Classes**: ham (4.825) e spam (747)
- **Balanceamento**: undersample — mantém 747 de cada classe (1.494 total)
- **Divisão**: 70% treino / 10% validação / 20% teste
- **Tokenização**: BPE do GPT-2, truncado em 128 tokens

### 3.4 Resultados esperados

| Métrica | Antes do fine-tuning | Depois (5 épocas) |
|---|---|---|
| Acurácia (treino) | ≈ 50 % | ≈ 99 % |
| Acurácia (validação) | ≈ 50 % | ≈ 96–98 % |
| Acurácia (teste) | ≈ 50 % | ≈ 95–99 % |
| F1-score (spam) | ≈ 0,00 | ≈ 0,95–0,99 |

A acurácia inicial ≈ 50 % é esperada: a `classification_head` tem pesos
aleatórios e o modelo adivinha. Após o fine-tuning o GPT-2 se torna um
classificador de spam muito preciso em apenas 5 épocas — graças ao
conhecimento linguístico trazido pelo pré-treino.

## 4. Entregável 4 — Avaliação do classificador

### 4.1 Métricas usadas

- **Acurácia**: `corretos / total` — funciona bem porque o dataset é balanceado.
- **Precisão**: `VP / (VP + FP)` — fração de spams identificados que são realmente spam.
- **Revocação**: `VP / (VP + FN)` — fração dos spams reais que foram identificados.
- **F1-score**: média harmônica de precisão e revocação — penaliza quem sacrifica uma para melhorar a outra.

### 4.2 Por que F1 importa em spam?

Errar um spam (FN) é irritante; classificar ham como spam (FP) pode fazer o
usuário perder e-mails importantes. O F1 equilibra esses dois tipos de erro.
Um classificador que chuta "ham" para tudo tem acurácia ≈ 87 % mas F1 = 0 —
o F1 expõe esse comportamento que a acurácia esconde.

## 5. Entregável 5 — Instruction Fine-Tuning

### 5.1 Diferença em relação à classificação

| | Classificação | Instruction FT |
|---|---|---|
| Cabeça de saída | `Linear(emb_dim, num_classes)` | `out_head` original (50.257 tokens) |
| Função de perda | cross-entropy sobre classes | cross-entropy sobre tokens |
| Dataset | rótulos binários | pares instrução/resposta |
| Parâmetros treináveis | só último bloco + cabeça | todos (ou LoRA) |
| O que o modelo aprende | "este texto é spam ou ham" | "dado este comando, gere esta resposta" |

### 5.2 Mascaramento do prompt

O modelo **não deve aprender a prever o prompt** — ele já sabe inglês e o
formato Alpaca. Só a resposta é informativa. Por isso os targets das
posições do prompt recebem `-100`, que o `F.cross_entropy` ignora
automaticamente com `ignore_index=-100`.

```
Tokens: [prompt ... prompt | resposta ... resposta]
Target: [ -100  ...  -100  |  token+1 ... token+N ]
```

### 5.3 Padding dinâmico (collate function)

As sequências de um mesmo lote têm comprimentos diferentes. A
`collate_instrucoes` padeia até o comprimento máximo **do lote** (não do
dataset inteiro), mantendo o padding mínimo necessário:

- `input_ids` recebem `pad_id=50256` (`<|endoftext|>`)
- `target_ids` das posições de padding recebem `-100` (ignoradas na perda)

### 5.4 Resultado esperado

Antes do fine-tuning o modelo gera texto fluente mas ignora a instrução.
Após 2 épocas no dataset Alpaca (1.100 exemplos), o modelo começa a seguir
os comandos — responde no formato esperado e aborda o tema da instrução.
Com mais épocas e um dataset maior (como o Alpaca original com 52.000
exemplos) a qualidade melhora muito mais.

## 6. Integração com as Sprints anteriores

```
Sprint 2: BPE tokenizer (tiktoken) ───────────────────┐
Sprint 2: GPTDatasetV1, create_dataloader_v1          │  reutilizados
Sprint 3: MultiHeadAttention (causal, multi-head)     │  sem alteração
Sprint 4: GPTModel (TransformerBlock × 12)            │
Sprint 5: generate, text_to_token_ids, get_device ────┘
                           ↓
Sprint 6: src/finetuning/
  pesos_pretrained.py  → carrega GPT-2 oficial no GPTModel
  classificador.py     → GPTClassifier (congela + classification_head)
  dados_cls.py         → SpamDataset (trunca, padeia, balanceia)
  instrucoes.py        → InstructionDataset (mascara prompt, padding dinâmico)
```

Nenhum módulo anterior foi modificado — o `src/finetuning/` é puramente
aditivo, seguindo o padrão cumulativo do projeto.