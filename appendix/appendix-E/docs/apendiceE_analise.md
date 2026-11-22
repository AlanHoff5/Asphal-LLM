# Apêndice E - Fine-tuning eficiente em parâmetros com LoRA

## 1. Objetivo

O Capítulo 6 adaptou o GPT-2 124M para classificar mensagens como spam ou
ham treinando o último Transformer Block, a LayerNorm final e uma cabeça de
classificação: 7.090.946 parâmetros, 4,35% do modelo. O Apêndice E faz a
mesma tarefa com **LoRA** (*Low-Rank Adaptation*): congela o modelo
inteiro e treina só duas matrizes pequenas por camada linear.

```text
GPT-2 124M pré-treinado (data/gpt2_pretrained.pth) + cabeça 768 → 2
  ↓  congela TODOS os parâmetros
  ↓  cada nn.Linear W vira  W·x + (alpha/rank)·x·A·B   (A, B treináveis; B = 0 no início)
Treino 5 épocas (AdamW) → avaliação → salvar só A e B → mesclar A·B em W
```

O código reutilizável fica em `src/finetuning/lora.py`:

| Função / classe | Papel |
|-----------------|-------|
| `LoRALayer` | o desvio `(alpha / rank) · x · A · B` |
| `LinearWithLoRA` | `nn.Linear` congelada + `LoRALayer` |
| `replace_linear_with_lora` | troca recursivamente as `nn.Linear` (com lista do que ignorar) |
| `preparar_classificador_lora` | congela o `GPTClassifier` e aplica o LoRA |
| `parametros_lora_esperados` | conta à mão dos parâmetros LoRA |
| `state_dict_lora` | só os tensores A e B, para salvar |
| `mesclar_lora` | incorpora A·B em W e devolve camadas `nn.Linear` comuns |

O dataset, a divisão (70/10/20, semente 123) e o laço de treino são os do
Capítulo 6 (`dados_cls.py` e `classificador.py`), para que a comparação
seja justa.

## 2. A ideia do LoRA

Num fine-tuning comum, uma camada de pesos W aprende uma atualização ΔW do
mesmo tamanho. O LoRA parte da observação de que essa atualização costuma
ter **posto baixo** e a escreve como o produto de duas matrizes estreitas:

```text
W' = W + (alpha / rank) · A · B        A: entrada × rank     B: rank × saída
```

Numa camada 768 × 768 com rank 16, ΔW teria 589.824 números; A e B juntos
têm 16 · (768 + 768) = 24.576, ou 4,2%. Duas escolhas de inicialização
fazem o método funcionar:

- **B = 0:** antes do treino, A·B = 0 e o modelo é exatamente o
  pré-treinado. O teste `test_lora_nao_muda_o_modelo_antes_do_treino`
  confirma: as saídas do classificador com e sem LoRA são iguais.
- **A aleatória** (Kaiming, como uma `nn.Linear`): se A também fosse zero,
  os gradientes de A e B seriam zero e nada aprenderia.

**Escala:** o código usa `alpha / rank`, como o repositório atual do autor.
A primeira edição do livro multiplicava só por `alpha`. Com `alpha / rank`
dá para mudar o rank sem reajustar a taxa de aprendizado, e por isso o lr
padrão é 8e-4, o mesmo do repositório do livro.

## 3. Onde o LoRA entra e quantos parâmetros treina

O `GPTClassifier` tem 74 camadas `nn.Linear`. O LoRA entra em 73:

| Camada | Por bloco | Formato | Parâmetros LoRA (rank 16) |
|--------|-----------|---------|---------------------------|
| `W_query`, `W_key`, `W_value`, `out_proj` | 4 | 768 → 768 | 4 × 24.576 = 98.304 |
| `ff.layers.0` | 1 | 768 → 3.072 | 61.440 |
| `ff.layers.2` | 1 | 3.072 → 768 | 61.440 |
| **Total por bloco** | | | **221.184** |
| 12 blocos | | | 2.654.208 |
| `classification_head` | | 768 → 2 | 12.320 |
| **Total** | | | **2.666.528** |

O total é **igual ao do livro** (2.666.528) e bate com a conta à mão de
`parametros_lora_esperados`.

A camada que fica de fora é o `gpt.out_head` (768 → 50.257). O livro troca
o `out_head` pela cabeça de 2 classes; o `GPTClassifier` do projeto mantém
o `out_head` e usa uma `classification_head` separada. Como ele não é usado
na classificação, colocar LoRA nele gastaria 16 · (768 + 50.257) = 816.400
parâmetros sem efeito nenhum. Por isso `preparar_classificador_lora` passa
`ignorar=("out_head",)`.

### Comparação com o Capítulo 6

| | Cap. 6 (último bloco) | Ap. E (LoRA, rank 16) |
|---|---:|---:|
| Parâmetros treináveis | 7.090.946 | 2.666.528 |
| Fração do modelo | 4,35% | 1,64% |
| O que é treinado | último bloco, LayerNorm final, cabeça | A e B em 73 camadas, todos os blocos |
| Arquivo do ajuste | modelo inteiro (~632 MB) | só A e B (10,2 MB) |

O LoRA treina 2,7 vezes menos parâmetros, mas espalhados pelos 12 blocos,
enquanto o Capítulo 6 só mexe no último.

### Efeito do rank

| Rank | Parâmetros treináveis | Fração do modelo |
|-----:|----------------------:|-----------------:|
| 4 | 666.632 | 0,41% |
| 8 | 1.333.264 | 0,82% |
| 16 | 2.666.528 | 1,64% |
| 32 | 5.333.056 | 3,27% |

O número de parâmetros cresce linearmente com o rank. Para medir o efeito
na acurácia, rode o script com `--rank 4 --alpha 4`, `--rank 8 --alpha 8`
e assim por diante: cada rank salva o próprio `lora_resultados_r<rank>.json`.

## 4. Resultados

Script: `experiments/lora_classificacao.py` → `resultados/lora_resultados.json`.

**Verificação feita:** o script e o notebook rodaram de ponta a ponta com
um GPT-2 reduzido e dados no formato do SMS Spam, porque o ambiente de
verificação não baixa o GPT-2 real. Isso confere o código: contagem de
parâmetros, treino só de A e B, salvamento e mesclagem. As acurácias reais
saem quando o script roda com o GPT-2 124M; preencha a tabela abaixo com
os valores do `lora_resultados.json`.

| Medida | Antes | Depois (5 épocas) | Referência do livro |
|--------|------:|------------------:|--------------------:|
| Acurácia de treino | | | ≈ 99,8% |
| Acurácia de validação | | | ≈ 98,0% |
| Acurácia de teste | | | ≈ 96,7% |
| F1 de teste | | | n/a |

Antes do treino a acurácia fica perto de 50%: a cabeça de classificação é
aleatória e, com B = 0, o LoRA ainda não muda nada. Os valores de
referência são do notebook atual do autor (GPT-2 124M, rank 16, lr 8e-4,
5 épocas) e servem como ordem de grandeza: a divisão do projeto e o
truncamento em 128 tokens não são idênticos aos do livro.

### Salvar e mesclar

- **Arquivo do LoRA:** `state_dict_lora` guarda só os 146 tensores A e B
  (73 camadas × 2): 10,2 MB, contra 632 MB do modelo inteiro em float32
  (1,6%). Para usar o classificador depois, basta o checkpoint
  pré-treinado, que já existe, e esse arquivo.
- **Mesclagem:** `mesclar_lora` soma `(alpha / rank) · (A·B)ᵀ` em cada W e
  devolve camadas `nn.Linear` comuns. Na verificação, a diferença máxima
  nos logits depois de mesclar foi de 4,8 × 10⁻⁷, ou seja, erro de
  arredondamento de float32. Depois da mescla, a inferência não tem custo
  extra.

## 5. Testes

`tests/test_lora.py` tem 12 testes:

| Grupo | O que verifica |
|-------|----------------|
| `LoRALayer` | formatos de A e B, B começa em zero, escala `alpha / rank` |
| `LinearWithLoRA` | no início a saída é igual à da `nn.Linear` |
| `replace_linear_with_lora` | troca todas as lineares (na `NeuralNetwork` do Apêndice A), respeita `ignorar`, 2.666.528 parâmetros no 124M |
| Congelamento | só tensores `.lora.A` e `.lora.B` são treináveis; o modelo com LoRA dá as mesmas saídas que o original |
| Treino | um passo do AdamW não mexe em nenhum peso congelado; o LoRA aprende uma tarefa simples até 100% |
| Salvar e mesclar | o arquivo tem só A e B e recarrega com `strict=False`; a mescla dá as mesmas saídas |

## 6. Limitações

- As acurácias com o GPT-2 real precisam ser medidas na máquina com GPU
  (seção 4).
- O LoRA ainda faz o forward e o backward pelo modelo inteiro. Ele economiza
  memória do **otimizador** (o AdamW guarda dois valores por parâmetro
  treinável), não a das ativações. Numa GPU de 4 GB, a vantagem aparece
  mais no tamanho do arquivo e na memória do AdamW do que no tempo.
- Com dropout de 0,1 (configuração do Capítulo 6, para comparar), os
  resultados variam um pouco entre execuções em GPU.

## 7. Onde isso aparece no projeto

| Peça | Vem de |
|------|--------|
| `GPTClassifier`, `treinar_classificador`, `avaliar_classificador` | Capítulo 6 (`src/finetuning/classificador.py`) |
| `SpamDataset` e divisão 70/10/20 | Capítulo 6 (`src/finetuning/dados_cls.py`) |
| Pesos do GPT-2 124M | Capítulo 6 (`carregar_pesos_pretrained.py`) |
| `requires_grad = False` e `nn.Module` | Apêndice A |
| `NeuralNetwork` usada nos testes de substituição | Apêndice A (`src/pytorch_basico/`) |
