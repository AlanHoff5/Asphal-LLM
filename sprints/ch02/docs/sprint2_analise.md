# Sprint 2 · Análise dos resultados

A Sprint 2 transforma texto em tensores que o modelo consegue receber. Este
documento resume a implementação, os experimentos e responde às 10 perguntas
da Seção 5 do roteiro usando os números obtidos.

Todos os experimentos usam o corpus do capítulo,
[`data/the-verdict.txt`](../../../data/the-verdict.txt): 20.479 caracteres,
4.690 tokens por palavra (vocabulário de 1.132) e 5.145 tokens BPE.

**Para reproduzir:**

```powershell
python -m pytest sprints/ch02/tests/ -s -v          # testes com os prints
python sprints/ch02/experiments/fluxo_sprint2.py    # fluxo de ponta a ponta
jupyter lab sprints/ch02/notebooks/ch02.ipynb       # experimentos (seção 2.9)
```

## 1. Pipeline implementado

| Etapa | Código | Saída |
|---|---|---|
| Tokenização | `tokenize()` em [`tokenizer.py`](../../../src/data/tokenizer.py) | lista de tokens (palavras e pontuação) |
| Vocabulário e Token IDs | `build_vocab()`, `SimpleTokenizerV1/V2` | `{token: id}` e o inverso; `<\|unk\|>` e `<\|endoftext\|>` |
| BPE | `get_bpe_tokenizer()` (`tiktoken`, GPT-2) | IDs de um vocabulário de 50.257 |
| Pares entrada/alvo | `GPTDatasetV1` em [`dataset.py`](../../../src/data/dataset.py) | janelas de `max_length`; alvo deslocado 1 posição |
| Lotes | `create_dataloader_v1()` em [`dataloader.py`](../../../src/data/dataloader.py) | `inputs` e `targets` de formato `(batch, L)` |
| Embeddings | `GPTInputEmbeddings` em [`embeddings.py`](../../../src/data/embeddings.py) | `token_emb + pos_emb`, formato `(batch, L, dim)` |

Exemplo de par entrada/alvo com BPE e `L = 4`:

```text
entrada: [  40,  367, 2885, 1464]
alvo:    [ 367, 2885, 1464, 1807]
```

## 2. Experimentos

Executados na seção 2.9 do [notebook](../notebooks/ch02.ipynb), salvo
indicação.

**Exp. 1 · Tokens por texto (BPE)**

| Texto | Caracteres | Tokens BPE | Caracteres por token |
|---|---:|---:|---:|
| Frase curta | 23 | 7 | 3,29 |
| Parágrafo | 500 | 123 | 4,07 |
| Texto completo | 20.479 | 5.145 | 3,98 |

**Exp. 2 · Palavras × BPE na mesma frase**
("In the sunlit terraces of the palace, Mrs. Gisburn said with pardonable pride.")

| Tokenizador | Vocabulário | Tokens |
|---|---:|---:|
| Por palavras | 1.132 | 16 |
| BPE | 50.257 | 21 |

**Exp. 3 · Tamanho do contexto × amostras** (stride = contexto, BPE)

| Contexto `L` | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amostras | 2.572 | 1.286 | 643 | 321 | 160 | 80 | 40 |

**Exp. 4 · Stride × sobreposição** (`L = 32`)

| Stride `S` | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|
| Amostras | 640 | 320 | 160 | 80 |
| Sobreposição | 75% | 50% | 0% | 0%, pula 32 tokens por janela |

**Exp. 5 · Tamanho do lote** (`L = 4`, 1.286 amostras, `drop_last=True`)

| `batch_size` | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| Lotes | 1.286 | 321 | 160 | 80 | 40 |
| Formato de `inputs` | (1, 4) | (4, 4) | (8, 4) | (16, 4) | (32, 4) |

**Exp. 6 · Dimensão do embedding** (vocabulário BPE, lote `(8, 4)`)

| `dim` | Saída | Parâmetros do token embedding | Parâmetros do positional embedding |
|---:|---|---:|---:|
| 8 | (8, 4, 8) | 402.056 | 32 |
| 64 | (8, 4, 64) | 3.216.448 | 256 |
| 256 | (8, 4, 256) | 12.865.792 | 1.024 |
| 768 | (8, 4, 768) | 38.597.376 | 3.072 |

**Exp. 7 · Custo de tokenização** (texto completo, média de 20 execuções):
BPE 2,4 ms; regex por palavras 3,3 ms.

**Exp. 8 · Efeito da posição:** o mesmo Token ID repetido nas posições 0 e 1
gera token embeddings **iguais**, mas entradas finais **diferentes**. A
distância entre as duas entradas (21,9) é exatamente a distância entre os
vetores de posição 0 e 1.

**Fluxo sobre o corpus** (`fluxo_sprint2.py`, tokenizador por palavras,
`L = S = 8`, `batch_size = 8`): 586 amostras, 73 lotes completos, entrada
final `(8, 8, 256)`.

## 3. Respostas às perguntas da Sprint

**1. Por que um LLM não pode trabalhar direto com o texto bruto?**
A rede só faz operações com tensores numéricos. O texto é uma sequência de
símbolos de tamanho variável. A tokenização o divide em unidades fixas e o
vocabulário dá um número a cada uma, que serve de índice para a camada de
embedding. No corpus, 20.479 caracteres viram 5.145 IDs (BPE).

**2. Qual é a função do vocabulário?**
É o dicionário fixo entre tokens e IDs, nos dois sentidos: `encode` leva o
texto ao modelo e `decode` traz a saída de volta a texto. Precisa ser o mesmo
no treino e na geração, senão um ID passaria a significar outro token. O
vocabulário por palavras tem 1.132 entradas e não conhece palavras fora do
corpus (viram `<|unk|>`). O do BPE tem 50.257 e nunca precisa de `<|unk|>`.

**3. Qual a diferença entre token e Token ID?**
Token é o pedaço de texto ("Gisburn", ","). Token ID é a posição dele no
vocabulário. O token tem significado; o ID é só um índice.

**4. Por que os Token IDs não servem como representação semântica?**
Os IDs seguem a ordem do vocabulário, não o significado: o ID 10 não é mais
parecido com o 11 do que com o 500. Se a rede recebesse os inteiros, trataria
essas diferenças numéricas como se fossem relações reais. Por isso o ID só
serve para buscar uma linha na tabela de embedding.

**5. Qual é a função dos embeddings?**
Trocar cada ID por um vetor denso e treinável. Os valores começam aleatórios e
são ajustados no treino, de modo que tokens usados de forma parecida fiquem
com vetores próximos. Com `dim = 256`, um lote `(8, 4)` vira `(8, 4, 256)`.

**6. Por que é necessário representar a posição dos tokens?**
O token embedding depende só do ID: o mesmo token em posições diferentes tem o
mesmo vetor (Exp. 8). A atenção da Sprint 3 também não sabe a ordem. Sem
posição, "gato persegue rato" e "rato persegue gato" teriam a mesma
representação. Somar o positional embedding torna as entradas diferentes
(distância de 21,9 no Exp. 8) sem mudar o formato do tensor.

**7. Qual a relação entre o tamanho do contexto e a quantidade de amostras?**
Com `N` tokens, contexto `L` e stride `S`, o `GPTDatasetV1` gera

$$\text{amostras} = \left\lceil \frac{N - L}{S} \right\rceil$$

O alvo precisa de um token a mais que a entrada, por isso a janela para em
`N − L`. Com stride igual ao contexto, dobrar o contexto corta as amostras
pela metade (Exp. 3: de 2.572 com `L = 2` para 40 com `L = 128`). Um stride
menor gera mais amostras, mas repetidas (Exp. 4: 640 amostras com 75% de
sobreposição). Um stride maior que o contexto desperdiça tokens.

**8. Qual o impacto da dimensão do embedding?**
Ela define o último eixo do tensor `(batch, L, dim)` e o tamanho da tabela de
embedding, `vocab × dim`. Com o vocabulário BPE, a tabela vai de 402 mil
parâmetros (`dim = 8`) para 38,6 milhões (`dim = 768`), crescimento linear
(Exp. 6). Mais dimensões permitem representações mais ricas, mas aumentam a
memória e o custo de todas as camadas seguintes, que trabalham com `dim`.

**9. Qual é a função do DataLoader?**
Agrupar as amostras do Dataset em lotes, embaralhá-las quando pedido e
descartar o último lote incompleto (`drop_last`). O `batch_size` muda só o
primeiro eixo (Exp. 5). No exemplo do fluxo, 5 amostras com `batch_size = 2`
dão 3 lotes, ou 2 com `drop_last=True`. O DataLoader não tokeniza nem calcula
embeddings: só organiza os pares que o Dataset já preparou.

**10. O que desta Sprint será usado pela atenção na Sprint 3?**
Os `input_embeddings` de formato `(batch, L, dim)`, que já carregam conteúdo e
posição. Os `targets` continuam como Token IDs e não entram na atenção: serão
usados no cálculo da perda, no treino (Cap. 5).

## 4. Conclusão

O pipeline da Sprint 2 produz lotes de entrada e alvo a partir de qualquer
texto e os converte em vetores com informação de conteúdo e de posição. Os
experimentos mostram os dois controles principais da preparação dos dados: o
contexto e o stride definem quantas amostras o texto rende, e a dimensão do
embedding define o tamanho da representação e o custo do modelo. O formato
`(batch, L, dim)` é a entrada da atenção na Sprint 3.