# Capítulo 6 e 7 — Fine-Tuning para Tarefas Específicas

> Sprint 6 do projeto **Asphal-LLM** · continua a partir do [Capítulo 5](../ch05/)

```mermaid
flowchart LR
  PW["Pesos\nPré-treinados\n6.1–6.4"] --> CLS["Classificação\nSpam/Ham\n6.5–6.7"]
  CLS --> EVAL["Avaliação\nAcurácia / F1\n6.8"]
  EVAL --> IFT["Instruction\nFine-Tuning\n7.2–7.5"]
  IFT --> CMP["Comparação\nAntes × Depois"]

  classDef done fill:#2d6a4f,color:#fff,stroke:none
  class PW,CLS,EVAL,IFT,CMP done
```

## Estrutura

```
ch06/
├── README.md
├── glossario/ch06.md                         # 17 termos (14 obrigatórios + 3)
├── docs/sprint6_analise.md                   # análise técnica dos entregáveis
├── experiments/
│   ├── carregar_pesos_pretrained.py          # E2: carrega GPT-2 e compara geração
│   ├── classificacao.py                      # E3+E4: fine-tuning spam + avaliação
│   ├── instrucoes.py                         # E5: instruction fine-tuning Alpaca
│   └── resultados/                           # JSON com métricas de cada script
└── tests/
    └── test_finetuning.py
```

O código reutilizável fica em [`src/finetuning/`](../../src/finetuning/):

| Arquivo | Conteúdo |
|---|---|
| `pesos_pretrained.py` | `carregar_pesos_gpt2`, `baixar_gpt2_hf`, `verificar_mapeamento` |
| `classificador.py` | `GPTClassifier`, `treinar_classificador`, `avaliar_classificador` |
| `dados_cls.py` | download, balanceamento, `SpamDataset`, `criar_loaders_spam` |
| `instrucoes.py` | formato Alpaca, `InstructionDataset`, `collate_instrucoes`, `treinar_instrucoes` |

## Como rodar

```bash
# Testes
python -m pytest sprints/ch06/tests/ -v

# E2 — carrega pesos do GPT-2 e salva checkpoint (~500 MB, requer internet)
python sprints/ch06/experiments/carregar_pesos_pretrained.py

# E3+E4 — fine-tuning spam/ham + acurácia e F1 (requer gpt2_pretrained.pth)
python sprints/ch06/experiments/classificacao.py

# E5 — instruction fine-tuning Alpaca (requer gpt2_pretrained.pth)
python sprints/ch06/experiments/instrucoes.py
```

> Pré-requisito: `pip install transformers`  
> Os checkpoints ficam em `data/` (fora do git — ~500 MB cada).

## Principais resultados

| Entregável | Resultado esperado |
|---|---|
| Pesos aleatórios vs pré-treinados | texto incoerente → inglês fluente |
| Acurácia antes do fine-tuning | ≈ 50 % (classification head aleatória) |
| Acurácia depois (5 épocas, spam) | ≈ 95–99 % no teste |
| F1-score (spam) | ≈ 0,95–0,99 |
| Instruction FT (antes) | resposta fora do contexto da instrução |
| Instruction FT (depois, 2 épocas) | resposta coerente com a instrução |

<details>
<summary>Estratégia de congelamento de camadas</summary>

No fine-tuning de classificação o `GPTClassifier` congela todos os
parâmetros do GPT e descongela apenas:

- Último(s) Transformer Block(s) (`n_blocos_treinaveis=1` por padrão)
- `final_norm` (LayerNorm final)
- `classification_head` (camada linear adicionada — sempre treinável)

Com `n_blocos_treinaveis=1` e o GPT-2 124M, apenas **~0,8 %** dos
parâmetros são treináveis. Isso reduz o custo de memória e evita
*catastrophic forgetting* — o modelo esquece menos o que aprendeu no
pré-treino.

Para fine-tuning completo use `n_blocos_treinaveis=-1`.

</details>

<details>
<summary>Formato Alpaca (instruction fine-tuning)</summary>

Cada exemplo do dataset tem três campos:

```json
{
  "instruction": "Rewrite the following sentence in the active voice.",
  "input": "The book was read by the student.",
  "output": "The student read the book."
}
```

O prompt formatado é:

```
Below is an instruction that describes a task, paired with an input
that provides further context. Write a response that appropriately
completes the request.

### Instruction:
Rewrite the following sentence in the active voice.

### Input:
The book was read by the student.

### Response:
The student read the book.
```

Os tokens do prompt recebem `target=-100` na cross-entropy, então o modelo
só aprende a prever os tokens da **resposta**.

</details>

## Resultados dos testes

| Suite | Testes |
|---|---|
| `test_finetuning.py` | 10 |
| **Total** | **10** |

Análise completa e discussão → [`docs/sprint6_analise.md`](docs/sprint6_analise.md)