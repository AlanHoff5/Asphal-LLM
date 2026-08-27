# Sprint 2 — Guia de Estudo (Capítulo 2: Working with Text Data)

> Objetivo deste documento: preparar para a **arguição individual**. O professor
> pode pedir para qualquer integrante *explicar um trecho de código*, *descrever
> um componente*, *interpretar uma equação*, *alterar um parâmetro* ou
> *interpretar um resultado experimental*. Este guia liga cada conceito do
> Capítulo 2 ao código do repositório e aos números dos nossos experimentos.

Leia junto com:
- `docs/sprint2_analise.md` — análise dos resultados
- `docs/sprint2_respostas.md` — respostas técnicas detalhadas
- `Build a Large Language Model (From Scratch) - Sebastian Raschka/technical glossary/Chapter 2.md` — glossário

---

## Índice

1. [O panorama: por que esta Sprint existe](#1-o-panorama)
2. [O pipeline completo, passo a passo](#2-o-pipeline-completo)
3. [Etapa 1 — Tokenização](#3-etapa-1--tokenização)
4. [Etapa 2 — Vocabulário e Token IDs](#4-etapa-2--vocabulário-e-token-ids)
5. [Etapa 3 — Byte Pair Encoding (BPE)](#5-etapa-3--byte-pair-encoding-bpe)
6. [Etapa 4 — Janela deslizante e pares entrada-alvo](#6-etapa-4--janela-deslizante-e-pares-entrada-alvo)
7. [Etapa 5 — Dataset e DataLoader](#7-etapa-5--dataset-e-dataloader)
8. [Etapa 6 — Token Embeddings](#8-etapa-6--token-embeddings)
9. [Etapa 7 — Positional Embeddings e Input Embedding](#9-etapa-7--positional-embeddings-e-input-embedding)
10. [Fórmulas e shapes que você precisa saber de cor](#10-fórmulas-e-shapes)
11. [Mapa código ↔ conceito](#11-mapa-código--conceito)
12. [Resultados dos experimentos (e o que provam)](#12-resultados-dos-experimentos)
13. [As 10 perguntas da análise — versão de bolso](#13-as-10-perguntas-da-análise)
14. [Perguntas prováveis de arguição + "explique este trecho"](#14-perguntas-prováveis-de-arguição)
15. [Pegadinhas e erros comuns](#15-pegadinhas-e-erros-comuns)
16. [Ponte para a Sprint 3 (Attention)](#16-ponte-para-a-sprint-3)
17. [Autoteste rápido](#17-autoteste-rápido)

---

## 1. O panorama

Uma rede neural só faz **operações numéricas sobre tensores** (somas, multiplicações
de matrizes, derivadas). Texto é uma sequência de caracteres de comprimento
variável — não dá para multiplicar "firewall" por uma matriz de pesos.

A Sprint 2 constrói a **camada de preparação de dados**: tudo o que acontece
*antes* do primeiro bloco Transformer. No fim, o texto vira um tensor
`(batch, contexto, dimensão)` de números reais treináveis. Nenhuma "inteligência"
ainda — só representação.

Frase-resumo para decorar:
> **A Sprint 2 transforma texto em um tensor `(B, T, D)` que carrega, para cada
> posição, "qual é o token" e "onde ele está".**

---

## 2. O pipeline completo

```
Texto bruto
   │  tokenize()            → divide em unidades (regex ou BPE)
   ▼
Tokens                      ["Firewall", "bloqueia", "ataques", "."]
   │  vocabulário (str→int)
   ▼
Token IDs                   [ 12, 45, 9, 3 ]           (inteiros = índices)
   │  janela deslizante (GPTDatasetV1)
   ▼
Pares (entrada, alvo)       entrada=[12,45,9]  alvo=[45,9,3]   (alvo = entrada +1)
   │  DataLoader
   ▼
Lote de Token IDs           tensor shape (B, T)        ex.: (8, 4)
   │  nn.Embedding (token)   + nn.Embedding (posição)
   ▼
Input Embeddings            tensor shape (B, T, D)     ex.: (8, 4, 256)
   ▼
→ entra no mecanismo de Attention (Sprint 3)
```

Cada seta é um componente que implementamos. `B` = batch size, `T` = context
length (nº de tokens por amostra), `D` = embedding dim.

---

## 3. Etapa 1 — Tokenização

**O que é:** dividir o texto em unidades mínimas (**tokens**). Um token pode ser
uma palavra, um pedaço de palavra ou um sinal de pontuação.

**Nossa implementação** (`src/data/tokenizer.py`):

```python
def tokenize(text):
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [token.strip() for token in tokens if token.strip()]
```

Como ler a regex `([,.:;?_!"()\']|--|\s)`:
- `[,.:;?_!"()\']` → qualquer um desses sinais de pontuação, **isoladamente**;
- `|--` → ou a sequência `--` (travessão);
- `|\s` → ou um espaço em branco (espaço, tab, quebra de linha).
- Os **parênteses de captura** fazem o `re.split` **manter** os separadores no
  resultado (a pontuação vira token, o espaço não).
- `token.strip() ... if token.strip()` remove os `''` e os espaços que sobram.

Exemplo:
```
"Firewall bloqueia ataques."  →  ["Firewall", "bloqueia", "ataques", "."]
```

**Pontos que caem na arguição:**
- Por que a pontuação é um token separado? Porque "ataques." e "ataques" devem
  ser a mesma palavra; separar o `.` evita duplicar entradas no vocabulário e dá
  ao modelo um sinal explícito de fim de frase.
- Essa tokenização é **case-sensitive** e **não faz stemming**: "Firewall" e
  "firewall" são tokens diferentes.
- É uma tokenização **baseada em regras** — simples, didática, mas frágil
  (qualquer palavra nova quebra ou vira `<|unk|>`). O GPT usa BPE (seção 5).

---

## 4. Etapa 2 — Vocabulário e Token IDs

**Vocabulário:** dicionário que mapeia cada token único → um inteiro (Token ID),
e vice-versa. É a "tabela de tradução" fixa entre texto e números.

```python
def build_vocab(text):
    unique_tokens = sorted(set(tokenize(text)))          # tokens únicos, ordenados
    for special_token in SPECIAL_TOKENS:                 # ("<|endoftext|>", "<|unk|>")
        if special_token not in unique_tokens:
            unique_tokens.append(special_token)
    return {token: index for index, token in enumerate(unique_tokens)}
```

- `set(...)` → remove repetições. `sorted(...)` → ordem determinística (o mesmo
  texto sempre gera o mesmo vocabulário).
- Os IDs são **0, 1, 2, ... N-1** — contíguos, atribuídos por ordem alfabética.
- **A ordem/valor do ID não tem significado semântico** (ver seção 13, pergunta 4).

**A tríade que o professor quer ouvir:**
> **Token ↔ Token ID ↔ Vocabulário**
> O token é a unidade textual. O Token ID é o número dele no vocabulário. O
> vocabulário é a tabela que liga os dois, nas duas direções.

**As duas classes** (`src/data/tokenizer.py`):

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {integer: token for token, integer in vocab.items()}

    def encode(self, text):                 # texto → Token IDs
        tokens = tokenize(text)
        return [self.str_to_int[token] for token in tokens]

    def decode(self, token_ids):            # Token IDs → texto
        text = " ".join(self.int_to_str[i] for i in token_ids)
        return re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)   # tira espaço antes de pontuação
```

- `V1` **quebra com `KeyError`** se aparecer um token fora do vocabulário.
- `V2` herda de `V1` e troca desconhecidos por `<|unk|>` antes de converter:

```python
class SimpleTokenizerV2(SimpleTokenizerV1):
    def encode(self, text):
        tokens = tokenize(text)
        tokens = [t if t in self.str_to_int else "<|unk|>" for t in tokens]
        return [self.str_to_int[t] for t in tokens]
```

**Tokens especiais** (`SPECIAL_TOKENS = ("<|endoftext|>", "<|unk|>")`):
| Token | Função |
|---|---|
| `<|unk|>` | representa qualquer palavra fora do vocabulário (só na V1/V2 por palavras; o BPE não precisa) |
| `<|endoftext|>` | marca a fronteira entre dois documentos concatenados; diz ao modelo "aqui terminou um texto e começou outro" |

Outros tokens especiais citados no livro (não usados por nós): `[BOS]` (início),
`[PAD]` (preenchimento para deixar sequências do mesmo tamanho). O GPT-2 usa só
`<|endoftext|>`, inclusive como padding.

**`decode(encode(x)) == x`?** Sim para tokens conhecidos. O `decode` recola com
espaços e remove o espaço antes da pontuação. Não recupera perfeitamente
espaçamentos originais estranhos (múltiplos espaços, quebras de linha).

---

## 5. Etapa 3 — Byte Pair Encoding (BPE)

**Problema da tokenização por palavras:** vocabulário gigante e ainda assim
incompleto — toda palavra nova vira `<|unk|>` e o modelo perde informação.

**BPE:** tokenização por **sub-palavras**. Começa com caracteres e vai **mesclando
os pares mais frequentes** até formar um vocabulário de tamanho fixo. Qualquer
palavra, mesmo inventada, se decompõe em sub-unidades conhecidas → **nunca precisa
de `<|unk|>`**.

No projeto usamos o BPE pronto do GPT-2 via `tiktoken`:

```python
def get_bpe_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")
```

Fatos do GPT-2 BPE:
- Vocabulário de **50.257** tokens.
- `<|endoftext|>` = ID **50256**.
- "Hello" = ID **15496** (uma palavra = um token).
- Palavras raras/estrangeiras quebram: `someunknownPlace` → vários tokens.
- Tende a gerar **mais tokens** por trecho do que a tokenização por palavras
  (ver Experimento 2: mesma frase = 16 tokens por palavra vs 21 por BPE).

**Quando o professor perguntar "por que BPE e não palavras?"**: robustez a
palavras novas, vocabulário de tamanho controlado, sem `<|unk|>`, e é o que o
modelo pré-treinado do GPT-2 espera (compatibilidade para as próximas Sprints).

---

## 6. Etapa 4 — Janela deslizante e pares entrada-alvo

**Tarefa de treino de um GPT:** *prever o próximo token* (Next-Word Prediction).
Para treinar, cada trecho de entrada precisa ter um **alvo** = o mesmo trecho
deslocado uma posição à direita.

```
sequência:  [40, 367, 2885, 1464, 1807]
entrada  x: [40, 367, 2885, 1464]
alvo     y:     [367, 2885, 1464, 1807]
```

Ou seja: dado `40` → prever `367`; dado `40,367` → prever `2885`; etc.

**Janela deslizante (sliding window):** percorre a sequência inteira de Token IDs
extraindo várias janelas. O **stride** é quanto a janela avança a cada passo.

`src/data/dataset.py`:

```python
for inicio in range(0, len(token_ids) - max_length, stride):
    entrada = token_ids[inicio : inicio + max_length]
    alvo    = token_ids[inicio + 1 : inicio + max_length + 1]
    self.input_ids.append(torch.tensor(entrada, dtype=torch.long))
    self.target_ids.append(torch.tensor(alvo,   dtype=torch.long))
```

- `max_length` = `T` = context size = quantos tokens por amostra.
- `stride` controla a **sobreposição** entre janelas:
  - `stride == max_length` → janelas **sem sobreposição** (cada token aparece em
    exatamente uma entrada);
  - `stride < max_length` → janelas **sobrepostas** → mais amostras, mas conteúdo
    repetido (risco de overfitting / desperdício);
  - `stride > max_length` → **pula** tokens (perde dados).
- `dtype=torch.long` é **obrigatório**: os IDs vão ser usados como **índices** de
  `nn.Embedding`, e índice tem que ser inteiro longo.

**Validações que colocamos** (e que o professor pode pedir para justificar):
```python
if max_length <= 0 or stride <= 0:  raise ValueError(...)
if len(token_ids) <= max_length:    raise ValueError("texto precisa ter mais tokens que max_length")
```
Sem pelo menos `max_length + 1` tokens não dá para formar nem uma entrada com seu alvo.

---

## 7. Etapa 5 — Dataset e DataLoader

### `GPTDatasetV1(torch.utils.data.Dataset)`

Implementa a interface que o PyTorch espera:
- `__len__()` → quantas amostras existem;
- `__getitem__(idx)` → devolve `(entrada, alvo)` da amostra `idx`.

No nosso código as janelas são **pré-computadas** no `__init__` e guardadas em
listas `self.input_ids` / `self.target_ids`. (Para corpora enormes o ideal seria
calcular sob demanda, mas para o objetivo didático tudo cabe na memória.)

### `create_dataloader_v1(...)` (`src/data/dataloader.py`)

```python
dataset = GPTDatasetV1(texto, tokenizer, max_length=max_length, stride=stride)
return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                  drop_last=drop_last, num_workers=num_workers)
```

Papel do **DataLoader**: pegar as amostras individuais do Dataset e **agrupá-las
em lotes (batches)** prontos para a rede. Ele **não** tokeniza, **não** cria
vocabulário, **não** calcula embeddings — só organiza.

Parâmetros:
| Parâmetro | O que faz | Efeito |
|---|---|---|
| `batch_size` | amostras por lote | muda só a **1ª dimensão** dos tensores: `(B, T)` |
| `shuffle` | embaralha a ordem das amostras a cada época | `True` no treino (evita o modelo aprender a ordem), `False` para reproduzir/depurar |
| `drop_last` | descarta o último lote se for incompleto | evita um lote de tamanho diferente que atrapalha estatísticas |
| `num_workers` | processos paralelos carregando dados | acelera; `0` = carrega no processo principal |

**Exemplo numérico** (dos nossos testes, `tests/test_dataloader.py`):
texto de 8 tokens, `max_length=3`, `stride=1` → **5 amostras**.
- `batch_size=2`, `drop_last=False` → **3 lotes** (2 + 2 + 1).
- `batch_size=2`, `drop_last=True` → **2 lotes** (2 + 2, descarta a 5ª amostra).

---

## 8. Etapa 6 — Token Embeddings

**Problema:** Token IDs são inteiros arbitrários. ID 10 não é "mais parecido" com
ID 11 do que com ID 500. Se a rede usasse o inteiro direto, interpretaria
diferenças numéricas sem sentido.

**Solução:** `torch.nn.Embedding(num_embeddings, embedding_dim)` — uma **tabela de
consulta (lookup table)** treinável. Cada ID indexa uma **linha** = um vetor denso
de `embedding_dim` números reais.

`src/data/embeddings.py`:
```python
self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
...
token_vectors = self.token_embedding(token_ids)   # (B, T) → (B, T, D)
```

- É **treinável**: os valores começam aleatórios (distribuição normal) e são
  ajustados por backpropagation para capturar relações úteis entre tokens.
- Nº de parâmetros da tabela = **`vocab_size × embedding_dim`**.
  Ex.: `50257 × 256 = 12 865 792` parâmetros só na camada de token embedding.
- É equivalente a "one-hot × matriz de pesos", mas implementado como lookup por
  eficiência (não multiplica a matriz inteira, só seleciona a linha).

**Shape:** entra `(B, T)` de inteiros, sai `(B, T, D)` de floats.

---

## 9. Etapa 7 — Positional Embeddings e Input Embedding

**Problema:** o mecanismo de Attention (Sprint 3) processa todos os tokens "de uma
vez" e **não tem noção de ordem**. Sem informação de posição:
```
"gato persegue rato"  ==  "rato persegue gato"   (para o modelo)
```
o que é obviamente errado.

**Solução:** uma **segunda** tabela de embedding, indexada pela **posição**
(0, 1, 2, ..., T-1) em vez do conteúdo. Usamos a variante **absoluta**, igual ao
GPT original.

`src/data/embeddings.py`:
```python
self.position_embedding = nn.Embedding(context_length, embedding_dim)
...
positions = torch.arange(token_ids.shape[1], device=token_ids.device)  # [0,1,...,T-1]
token_vectors    = self.token_embedding(token_ids)      # (B, T, D)
position_vectors = self.position_embedding(positions)   # (T, D)
return token_vectors + position_vectors                 # (B, T, D)  via broadcasting
```

**Input Embedding = Token Embedding + Positional Embedding.**

- A soma funciona por **broadcasting**: `(B, T, D) + (T, D)` → o PyTorch repete o
  bloco de posições para cada item do lote.
- O shape **não muda**: continua `(B, T, D)`.
- Cada vetor final responde a duas perguntas: **"qual token?"** (parte do token
  embedding) e **"em que posição?"** (parte do positional embedding).

**Demonstração experimental** (Experimento 8 do notebook / `test_posicao_altera_a_representacao`):
> Colocamos o **mesmo Token ID** nas 4 posições. O token embedding puro é
> **idêntico** nas 4 linhas. A entrada final (`token + posição`) é **diferente**
> em cada linha, e a diferença entre a posição 0 e a 1 é exatamente
> `pos_embedding[0] - pos_embedding[1]`. → **a posição altera a representação.**

- `context_length` na tabela de posição é o **limite de contexto** do modelo:
  não dá para processar sequência maior que isso (nosso `forward` valida e
  levanta `ValueError`).

---

## 10. Fórmulas e shapes

**Nº de amostras produzidas pelo `GPTDatasetV1`** (com `N` = total de Token IDs,
`L` = `max_length`, `S` = `stride`, e `N > L`):

$$
\text{nº de amostras} = \left\lceil \frac{N - L}{S} \right\rceil
= \left\lfloor \frac{N - L - 1}{S} \right\rfloor + 1
$$

(é a contagem de `range(0, N - L, S)`).

Consequências:
- **contexto maior (`L`↑) → menos amostras** (o mesmo corpus "cabe" menos vezes);
- **stride menor (`S`↓) → mais amostras**, com sobreposição;
- `S = L` → cobertura exata, sem sobreposição nem buraco.

**Nº de lotes:**
$$
\text{nº de lotes} =
\begin{cases}
\left\lfloor \dfrac{\text{amostras}}{B} \right\rfloor & \text{se } \texttt{drop\_last=True}\\[2ex]
\left\lceil \dfrac{\text{amostras}}{B} \right\rceil & \text{se } \texttt{drop\_last=False}
\end{cases}
$$

**Nº de parâmetros das camadas de embedding:**
- token embedding: `vocab_size × embedding_dim`
- positional embedding: `context_length × embedding_dim` (bem pequeno)

**Shapes ao longo do pipeline** (exemplo `B=8`, `T=4`, `D=256`):
| Estágio | Shape | Tipo |
|---|---|---|
| Lote de Token IDs (saída do DataLoader) | `(8, 4)` | `int64` |
| Token embeddings | `(8, 4, 256)` | `float32` |
| Positional embeddings | `(4, 256)` | `float32` |
| Input embeddings (soma) | `(8, 4, 256)` | `float32` |
| Alvos (`targets`) | `(8, 4)` | `int64` |

---

## 11. Mapa código ↔ conceito

| Conceito do Capítulo 2 | Arquivo | Símbolo |
|---|---|---|
| Tokenização por regras | `src/data/tokenizer.py` | `tokenize()` |
| Leitura do corpus | `src/data/tokenizer.py` | `load_text()` |
| Construção do vocabulário | `src/data/tokenizer.py` | `build_vocab()` |
| Token ↔ ID (encode/decode) | `src/data/tokenizer.py` | `SimpleTokenizerV1` |
| Tratamento de desconhecidos | `src/data/tokenizer.py` | `SimpleTokenizerV2` (`<|unk|>`) |
| BPE (GPT-2) | `src/data/tokenizer.py` | `get_bpe_tokenizer()` (`tiktoken`) |
| Janela deslizante + pares entrada-alvo | `src/data/dataset.py` | `GPTDatasetV1` |
| Lotes de treino | `src/data/dataloader.py` | `create_dataloader_v1()` |
| Token embedding | `src/data/embeddings.py` | `GPTInputEmbeddings.token_embedding` |
| Positional embedding | `src/data/embeddings.py` | `GPTInputEmbeddings.position_embedding` |
| Input embedding (soma) | `src/data/embeddings.py` | `GPTInputEmbeddings.forward()` |
| Fluxo ponta a ponta | `experiments/fluxo_sprint2.py` | `mostrar_fluxo()` |
| Experimentos | `notebooks/InteligenciaArtificial_SistemasInteligentes.ipynb` | seção 2.9 |
| Testes | `tests/` | 17 testes pytest |

---

## 12. Resultados dos experimentos

Corpus principal: `the-verdict.txt` — **20 479 caracteres**, **4 690** tokens por
palavra, vocabulário de **1 130** tokens únicos (+2 especiais = **1 132**),
**5 145** tokens BPE.

Corpus do projeto: `data/comptia_security_pluse_701/cleaned_data.txt` —
**682 480 caracteres**, **110 334** tokens por palavra, vocabulário de **9 291**.

### Exp. 1 — Tokens por texto (BPE)
| Texto | Caracteres | Tokens BPE | chars/token |
|---|---|---|---|
| Frase curta | 23 | 7 | 3,29 |
| Parágrafo médio | 500 | 123 | 4,07 |
| Texto completo | 20 479 | 5 145 | 3,98 |

→ a razão caracteres/token estabiliza em ~4 para texto em inglês corrido.

### Exp. 2 — Palavras vs BPE (mesma frase)
`"In the sunlit terraces of the palace, Mrs. Gisburn said with pardonable pride."`
- por palavras (vocab 1 132): **16 tokens**
- BPE (vocab 50 257): **21 tokens**

→ BPE tem vocabulário ~44× maior mas gera **mais** tokens por frase (quebra
palavras compostas/raras em pedaços).

### Exp. 3 — Context size × nº de amostras (stride = context, `the-verdict` BPE)
| context | amostras |
|---|---|
| 2 | 2 572 |
| 4 | 1 286 |
| 8 | 643 |
| 16 | 321 |
| 32 | 160 |
| 64 | 80 |
| 128 | 40 |

→ dobrar o contexto ≈ **metade** das amostras. Confere com `⌈(N−L)/S⌉`.

### Exp. 4 — Stride × sobreposição (context = 32)
| stride | amostras | sobreposição |
|---|---|---|
| 8 | 640 | 75 % |
| 16 | 320 | 50 % |
| 32 | 160 | 0 % |
| 64 | 80 | 0 % (pula tokens) |

### Exp. 5 — Batch size (context = 4)
`batch_size` muda só a 1ª dimensão: `(1,4) → (4,4) → (8,4) → (16,4) → (32,4)`.
Nº de lotes = `amostras / batch_size`.

### Exp. 6 — Dimensão do embedding (vocab BPE = 50 257)
| `output_dim` | shape input_emb | params token_emb |
|---|---|---|
| 8 | (8, 4, 8) | 402 056 |
| 64 | (8, 4, 64) | 3 216 448 |
| 256 | (8, 4, 256) | 12 865 792 |
| 768 | (8, 4, 768) | 38 597 376 |

→ nº de parâmetros cresce **linearmente** com `output_dim` (= `vocab × dim`).
Mais dimensão = mais capacidade de representação, mais memória e mais custo em
**todas** as camadas seguintes (inclusive Attention).

### Exp. 7 — Custo de tokenização
BPE (`tiktoken`, núcleo em Rust) ≈ **1,6 ms** para o texto todo;
regex por palavras ≈ **2,8 ms**. BPE é mais sofisticado **e** mais rápido.

---

## 13. As 10 perguntas da análise

Respostas curtas para revisão (versão longa em `docs/sprint2_respostas.md`).

1. **Por que um LLM não trabalha com texto bruto?**
   Rede neural = operações sobre tensores numéricos. Texto tem símbolos discretos
   e comprimento variável; precisa virar Token IDs → tensores.

2. **Função do vocabulário?**
   Mapear cada token conhecido a um ID (e o inverso). Precisa ser **fixo** entre
   treino e geração — se um ID mudar de token, o modelo perde o que aprendeu.

3. **Token vs Token ID?**
   Token = unidade textual (palavra/subpalavra/pontuação). Token ID = o número
   dele no vocabulário. O ID é um **índice**, não um significado.

4. **Por que não usar Token IDs como representação semântica?**
   A distância entre inteiros é arbitrária (ID 10 não é "perto" de 11). O
   embedding aprende vetores em que a proximidade **sim** reflete relação.

5. **Função dos embeddings?**
   Converter cada ID discreto em um vetor denso de dimensão fixa, **treinável**,
   que a rede consegue processar e ajustar.

6. **Por que representar a posição?**
   A Attention não tem noção de ordem. Sem positional embedding, permutar tokens
   não mudaria a saída. A posição é somada ao token embedding.

7. **Relação contexto × nº de amostras?**
   `amostras ≈ ⌈(N − L) / S⌉`. Contexto maior → menos amostras; stride menor →
   mais amostras (com sobreposição).

8. **Impacto da dimensão do embedding?**
   Tabela de token embedding tem `vocab × dim` parâmetros. `dim` maior →
   representação mais rica, mas mais memória/computação em toda a arquitetura.
   Muda o último eixo de `(B, T, D)`.

9. **Função do DataLoader?**
   Agrupar as amostras do Dataset em lotes, embaralhar, descartar lote incompleto,
   paralelizar o carregamento. Não tokeniza nem cria embeddings.

10. **O que a Sprint 2 entrega para a Attention (Sprint 3)?**
    - `inputs`: Token IDs em lotes → viram `input_embeddings` de shape `(B, T, D)`;
    - `targets`: Token IDs deslocados, guardados como inteiros para a função de
      perda (não entram no cálculo da atenção).

---

## 14. Perguntas prováveis de arguição

**"Explique esta linha: `re.split(r'([,.:;?_!"()\']|--|\s)', text)`"**
→ ver seção 3. Ponto-chave: grupo de captura mantém os separadores; depois
filtramos os vazios.

**"O que acontece se eu passar `stride=1` e `max_length=256` num texto de 300 tokens?"**
→ `range(0, 300−256, 1)` = `range(0, 44)` → **44 amostras**, altamente
sobrepostas (cada uma difere da anterior por 1 token).

**"Por que `dtype=torch.long` nos tensores do Dataset?"**
→ porque serão índices de `nn.Embedding`, e indexação exige inteiro (int64).

**"Qual a diferença entre `token_embedding` e `position_embedding`?"**
→ mesma classe (`nn.Embedding`), papéis diferentes: uma é indexada pelo **ID do
token** (tamanho `vocab_size`), a outra pela **posição** (tamanho
`context_length`). Ambas treináveis. Somadas formam o input embedding.

**"Mostre que a soma não muda o shape."**
→ `(B, T, D) + (T, D)` → broadcasting repete `(T, D)` em cada item do batch →
resultado `(B, T, D)`.

**"Se eu aumentar `output_dim` de 256 para 512, o que muda?"**
→ params do token embedding dobram (`vocab × 512`); `input_embeddings` vira
`(B, T, 512)`; todas as camadas seguintes ficam mais caras. Nada muda em `B` e `T`.

**"Por que o alvo é a entrada deslocada em 1?"**
→ tarefa de Next-Word Prediction: para cada posição `i` da entrada, o alvo é o
token `i+1` da sequência original. É assim que o modelo aprende a continuar texto.

**"Onde entra o `<|endoftext|>` no treino real?"**
→ entre documentos diferentes, antes de tokenizar, para o modelo aprender a
fronteira e não "misturar" textos.

**"Altere o código para gerar janelas sem sobreposição."**
→ passar `stride = max_length` no `create_dataloader_v1` / `GPTDatasetV1`.

**"O que é `num_workers` e quando aumentar?"**
→ processos paralelos de carregamento; aumentar quando o carregamento/pré-proc.
for gargalo (não é o nosso caso com dados na memória).

---

## 15. Pegadinhas e erros comuns

- **`SimpleTokenizerV1` quebra com `KeyError`** em palavra nova. É de propósito —
  motiva a V2 e os tokens especiais. (No notebook, a célula que provoca isso está
  num `try/except`.)
- **Vocabulário diferente por texto:** `create_tokenizer(texto)` constrói um
  vocabulário **novo** para aquele texto. Dois textos → dois vocabulários
  incompatíveis. Num modelo real o vocabulário é **um só**, salvo em disco.
- **`decode` não é inverso perfeito de `encode`** para espaçamento/quebras de linha.
- **`context_length` do positional embedding = limite rígido.** Sequência maior
  que isso → `ValueError` no nosso `forward` (e IndexError no PyTorch cru).
- **Confundir "nº de tokens" com "nº de amostras".** Um texto de 5 145 tokens com
  contexto 4 e stride 4 gera **1 286 amostras**, não 5 145.
- **BPE gera mais tokens, não menos.** Vocabulário maior ≠ menos tokens por frase.
- **`shuffle=True` no teste** deixa a saída não-determinística — usamos `False`
  nos testes para poder verificar valores exatos.
- **A Sprint 2 não gera texto.** O pipeline termina num tensor `(B, T, D)`. Geração
  é Sprint 4+.

---

## 16. Ponte para a Sprint 3

O que a Sprint 3 (Attention) **recebe** desta Sprint:

```
input_embeddings : (B, T, D)     ← token embedding + positional embedding
targets          : (B, T)        ← Token IDs deslocados (só para a loss, depois)
```

Por que a Attention precisa exatamente disso:
- ela calcula **relações entre as posições da mesma sequência** (quem "presta
  atenção" em quem) — opera sobre o eixo `T`, com vetores de dimensão `D`;
- precisa do **positional embedding** já somado, senão não distingue ordem;
- `D` (a dimensão do embedding) vira a dimensão sobre a qual são calculados
  Query, Key e Value.

Frase para fechar a arguição:
> "A Sprint 2 entrega um tensor `(batch, contexto, dimensão)` onde cada vetor já
> carrega conteúdo + posição. A Sprint 3 usa esse tensor para calcular, via
> self-attention, o quanto cada token deve influenciar a representação dos outros."

---

## 17. Autoteste rápido

Tente responder sem olhar (respostas nas seções indicadas):

1. Escreva o pipeline completo, de "Texto" até "entrada do modelo". *(§2)*
2. O que a regex de `tokenize()` captura e por que os parênteses importam? *(§3)*
3. Explique "Token ↔ Token ID ↔ Vocabulário" em uma frase. *(§4)*
4. Cite duas vantagens do BPE sobre a tokenização por palavras. *(§5)*
5. Dado `[10, 11, 12, 13, 14]` e contexto 3, quais são a 1ª entrada e o 1º alvo? *(§6)*
6. Um texto tem 1 000 tokens, contexto 50, stride 25. Quantas amostras? *(§10 — `⌈950/25⌉ = 38`)*
7. `batch_size=16`, contexto 8, `output_dim=128`. Shape do `input_embeddings`? *(§10 — `(16, 8, 128)`)*
8. Por que Token IDs não servem como representação semântica? *(§13.4)*
9. Prove, com um experimento, que a posição altera a representação. *(§9)*
10. Quantos parâmetros tem a tabela de token embedding com vocab 50 257 e dim 256? *(§10 — 12 865 792)*
11. O que muda no pipeline se eu passar `drop_last=True`? *(§7)*
12. O que a Sprint 3 recebe da Sprint 2, e em que shape? *(§16)*

---

### Comandos para revisar na prática

```bash
# fluxo completo comentado, com números
python experiments/fluxo_sprint2.py

# todos os testes com os prints explicativos
python -m pytest tests/ -s -v

# experimentos variando parâmetros
# abrir notebooks/InteligenciaArtificial_SistemasInteligentes.ipynb (seção 2.9)
```
