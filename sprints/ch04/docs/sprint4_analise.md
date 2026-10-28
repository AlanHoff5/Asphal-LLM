# Sprint 4 - Implementando o modelo GPT

## 1. Objetivo

A Sprint 4 junta tudo o que foi construído até aqui em um modelo que
recebe uma sequência de Token IDs e devolve, para cada posição, uma
pontuação (logit) para cada token do vocabulário ser o próximo. A partir
desses logits, o modelo gera texto de forma autoregressiva.

```text
Token IDs                                   (batch, num_tokens)
  ↓
Token + Positional Embedding  (Sprint 2)    (batch, num_tokens, emb_dim)
  ↓
Dropout
  ↓
Transformer Block × n_layers  (Sprint 3 + 4)
  │  LayerNorm → Multi-Head Attention → Dropout → + atalho
  │  LayerNorm → FeedForward (GELU)   → Dropout → + atalho
  ↓                                         (batch, num_tokens, emb_dim)
LayerNorm final
  ↓
Camada de saída (LM head)                   (batch, num_tokens, vocab_size) = LOGITS
  ↓
Geração: última posição → softmax → seleção → anexa → repete
```

Todo o código fica em `src/model/`, reaproveitando `src/data/embeddings.py`
(Sprint 2) e `src/attention/multi_head_attention.py` (Sprint 3) sem
alterações. Duas configurações são usadas:

| Configuração | vocab_size | context_length | emb_dim | n_heads | n_layers | Parâmetros |
|--------------|-----------:|---------------:|--------:|--------:|---------:|-----------:|
| `GPT_CONFIG_124M` (GPT-2 small, livro) | 50.257 | 1.024 | 768 | 12 | 12 | 163.009.536 |
| `GPT_CONFIG_MINI` (projeto) | 50.257 | 128 | 128 | 4 | 4 | 13.673.984 |

A configuração MINI mantém a mesma estrutura e o mesmo tokenizador do GPT-2,
mas cabe em segundos de CPU; é a base dos experimentos.

## 2. Entregável 2 - Transformer Block

### 2.1 Componentes e função de cada um

| Componente | Arquivo | Função no bloco |
|------------|---------|-----------------|
| `LayerNorm` | `layer_norm.py` | Deixa o vetor de cada token com média 0 e variância 1 antes de cada sub-camada, mantendo as ativações numa escala estável. `scale` e `shift` treináveis permitem ao treino desfazer a normalização se for útil. |
| `MultiHeadAttention` | `src/attention/` (Sprint 3) | Mistura informação **entre** tokens: cada posição combina os vetores das posições anteriores (máscara causal), com `n_heads` padrões de relação em paralelo. |
| `GELU` | `feed_forward.py` | Não linearidade suave; não zera entradas negativas, então o gradiente não morre como na ReLU. |
| `FeedForward` | `feed_forward.py` | Processa a informação **dentro** de cada token: expande para `4·emb_dim`, aplica GELU e contrai de volta. Mesmos pesos para todas as posições. |
| `Dropout` (`drop_shortcut`) | `transformer_block.py` | Zera unidades aleatórias no treino, antes de somar o atalho; desligado em `model.eval()`. |
| Conexões de atalho | `transformer_block.py` | `x + subcamada(x)`: caminho curto para o gradiente e preservação da informação original do token. |

A ordem é a do GPT-2 (**Pre-LayerNorm**): a normalização vem antes de cada
sub-camada, e não depois da soma como no Transformer original
(Post-LayerNorm). O caminho do atalho fica sem nenhuma operação, o que
deixa o treino de redes profundas mais estável.

### 2.2 Dimensões dos tensores dentro de um bloco

Para o lote do livro (2 frases × 4 tokens) no GPT-2 small (`emb_dim=768`,
`n_heads=12`, `head_dim=64`):

| Passo | Operação | Shape |
|-------|----------|-------|
| Entrada do bloco | | `(2, 4, 768)` |
| `norm1` | LayerNorm por token | `(2, 4, 768)` |
| `att`: Q, K, V | 3 × `Linear(768, 768)` | `(2, 4, 768)` cada |
| `att`: divisão em cabeças | `view` + `transpose` | `(2, 12, 4, 64)` |
| `att`: scores | `Q @ Kᵀ` + máscara causal | `(2, 12, 4, 4)` |
| `att`: pesos | softmax(scores / √64) | `(2, 12, 4, 4)` |
| `att`: contexto | `pesos @ V`, junta cabeças | `(2, 4, 768)` |
| `att`: `out_proj` | `Linear(768, 768)` | `(2, 4, 768)` |
| dropout + atalho | `x + atalho` | `(2, 4, 768)` |
| `norm2` | LayerNorm por token | `(2, 4, 768)` |
| `ff.layers[0]` | `Linear(768, 3072)` | `(2, 4, 3072)` |
| `ff.layers[1]` | GELU | `(2, 4, 3072)` |
| `ff.layers[2]` | `Linear(3072, 768)` | `(2, 4, 768)` |
| dropout + atalho | `x + atalho` | `(2, 4, 768)` |
| **Saída do bloco** | | **`(2, 4, 768)`** |

O shape de saída é igual ao de entrada: é isso que permite somar os atalhos
e empilhar `n_layers` blocos. O único tensor que cresce com o quadrado do
contexto é a matriz de atenção `(batch, n_heads, n, n)`.

### 2.3 Parâmetros de um bloco (GPT-2 small)

| Parte | Cálculo | Parâmetros |
|-------|---------|-----------:|
| Q, K, V (sem bias) | 3 × 768 × 768 | 1.769.472 |
| `out_proj` | 768 × 768 + 768 | 590.592 |
| **Atenção** | | **2.360.064** |
| `Linear(768, 3072)` | 768 × 3072 + 3072 | 2.362.368 |
| `Linear(3072, 768)` | 3072 × 768 + 768 | 2.360.064 |
| **FeedForward** | | **4.722.432** |
| 2 LayerNorm | 2 × (768 + 768) | 3.072 |
| **Total do bloco** | | **7.085.568** |

A FeedForward tem o dobro de parâmetros da atenção (Exercício 4.1). Todos
esses valores são conferidos em `tests/test_transformer_block.py`.

## 3. Entregável 3 - Modelo GPT

`GPTModel` (`src/model/gpt_model.py`) organiza os componentes em quatro
etapas, que aparecem com esses nomes no código:

1. `embeddings` - `GPTInputEmbeddings` da Sprint 2: `token_embedding`
   (50.257 × 768) + `position_embedding` (1.024 × 768), somados por
   broadcasting. Ele já recusa sequências maiores que `context_length`.
2. `drop_emb` - dropout sobre os embeddings.
3. `trf_blocks` - `nn.Sequential` com `n_layers` blocos **independentes**
   (cada um com seus pesos).
4. `final_norm` + `out_head` - a LayerNorm final é necessária porque, no
   Pre-LN, a saída do último bloco ainda não foi normalizada; `out_head` é
   `Linear(768, 50257)` sem bias e produz os logits.

### 3.1 Onde estão os parâmetros (GPT-2 small)

| Componente | Parâmetros | % |
|------------|-----------:|--:|
| Token embedding (50.257 × 768) | 38.597.376 | 23,7% |
| Positional embedding (1.024 × 768) | 786.432 | 0,5% |
| 12 Transformer Blocks | 85.026.816 | 52,2% |
| LayerNorm final | 1.536 | 0,0% |
| Camada de saída (768 × 50.257) | 38.597.376 | 23,7% |
| **Total** | **163.009.536** | |

O total dá 163M e não 124M porque o GPT-2 original usa **weight tying**: a
mesma matriz no token embedding e na camada de saída (as duas têm shape
50.257 × 768). Com `weight_tying=True`, o nosso modelo passa a ter
124.412.160 parâmetros. O livro mantém as matrizes separadas no Cap. 4, e
nós também (o padrão é `False`). Só os pesos ocupam 621,83 MB em float32.

### 3.2 Validação contra o livro

Mesmo usando o `GPTInputEmbeddings` e a `MultiHeadAttention` das Sprints
anteriores, o modelo reproduz **exatamente** os números publicados no
livro com `torch.manual_seed(123)` (notebook `ch04.ipynb`):

- logits do lote `"Every effort moves you"` / `"Every day holds a"`
  começando em `0.3613, 0.4222, -0.0711`;
- geração gulosa de 6 tokens a partir de `"Hello, I am"`:
  `[15496, 11, 314, 716, 27018, 24086, 47843, 30961, 42348, 7267]` →
  `"Hello, I am Featureiman Byeswickattribute argue"`.

Isso só acontece se todas as peças criam os pesos na mesma ordem e fazem as
mesmas contas que o código do livro, o que confirma a integração entre as
Sprints 2, 3 e 4.

### 3.3 Contagem analítica

`parametros_esperados(cfg)` calcula a tabela da seção 3.1 a partir da configuração,
sem instanciar o modelo; os testes conferem que o cálculo bate com
`sum(p.numel())` em várias configurações.

## 4. Entregável 4 - Inferência e geração de texto

### 4.1 De logits a texto

| Passo | Código (`generation.py`) | Shape |
|-------|--------------------------|-------|
| Contexto atual | `idx` | `(batch, n)` |
| 1. Corta o contexto | `idx[:, -context_size:]` | `(batch, ≤ context_size)` |
| 2. Forward sem gradiente | `with torch.no_grad(): model(idx_cond)` | `(batch, n, vocab)` |
| 3. Só a última posição | `logits[:, -1, :]` | `(batch, vocab)` |
| 4. Probabilidades | `torch.softmax(logits, dim=-1)` | `(batch, vocab)` |
| 5. Seleção | `argmax` (gulosa) ou `multinomial` (sorteio) | `(batch, 1)` |
| 6. Anexa e repete | `torch.cat((idx, idx_next), dim=1)` | `(batch, n + 1)` |

- **Logits** são pontuações brutas (podem ser negativas e não somam 1). O
  **softmax** os transforma em probabilidades sem mudar a ordem; por isso,
  na decodificação gulosa ele é dispensável (o maior logit já é a maior
  probabilidade).
- Só a **última posição** interessa porque, com a máscara causal, ela é a
  única que viu o texto inteiro; a posição *t* prevê o token *t + 1*.
- O **corte do contexto** é obrigatório: o Positional Embedding só tem
  `context_length` posições (o teste `test_contexto_e_cortado_quando_passa_do_limite`
  mostra que sem ele o modelo levanta erro).
- **`model.eval()`** desliga o dropout; sem isso a mesma entrada daria
  saídas diferentes a cada chamada. **`torch.no_grad()`** evita guardar o
  grafo de gradientes, economizando memória.
- **Autorregressão**: cada token escolhido vira entrada do passo seguinte,
  então gerar *k* tokens exige *k* forwards completos.

### 4.2 Estratégias de seleção

`generate_text_simple` reproduz a decodificação **gulosa** do livro. A
função `generate` acrescenta as estratégias que o livro apresenta na seção
5.3 e que o roteiro da Sprint 4 pede:

- **Temperatura** (`temperature > 0`): `softmax(logits / T)` seguido de
  sorteio com `torch.multinomial`. `T < 1` concentra a probabilidade nos
  favoritos; `T > 1` achata a distribuição. `T = 0` é tratado como gulosa.
- **Top-k**: antes do softmax, os logits fora dos *k* maiores viram `-inf`
  (probabilidade 0). Limita o sorteio aos candidatos plausíveis mesmo com
  temperatura alta. `top_k = 1` equivale à gulosa.

Com os logits de exemplo `[2, 1, 0]`, os testes conferem os valores do
glossário: T=1 → `[0,665; 0,245; 0,090]`, T=0,5 → `[0,867; 0,117; 0,016]`,
T=2 → `[0,506; 0,307; 0,186]` e top-k=2 → `[0,731; 0,269; 0]`.

## 5. Entregável 5 - Experimentação e análise

Todos os experimentos estão em `experiments/fluxo_sprint4.py` e os números
completos em `experiments/resultados/sprint4_resultados.json`. Salvo
indicação, a base é a `GPT_CONFIG_MINI` com um lote de 4 sequências de 64
tokens, medido na CPU (2 threads). O tempo é o **menor** de 7 forwards
após aquecimento; os valores absolutos mudam de computador para
computador, mas as tendências se mantêm. Gráficos gerados:
`gradientes_atalho.png`, `custo_arquitetura.png` e `temperatura.png`.

### Experimento 1 - Conexões de atalho e vanishing gradient

**Configuração:** rede da seção 4.4 (camadas `Linear(3, 3) + GELU`,
última `Linear(3, 1)`), entrada `[1, 0, -1]`, alvo 0, perda MSE. Medimos o
gradiente médio `|dL/dW|` de cada camada com 5, 10 e 20 camadas.

| Rede | 1ª camada sem atalho | 1ª camada com atalho | Última / 1ª sem atalho | Última / 1ª com atalho |
|------|---------------------:|---------------------:|-----------------------:|-----------------------:|
| 5 camadas | 0,000202 | 0,2217 | 25× | 6× |
| 10 camadas | ≈ 3 × 10⁻⁷ | 0,5696 | ≈ 24.600× | 13× |
| 20 camadas | ≈ 4 × 10⁻¹⁶ | 3,4452 | ≈ 1,6 × 10¹³× | 24× |

**Análise:** com 5 camadas reproduzimos exatamente o livro (0,0002 sem
atalho, 0,22 com atalho). Ao aumentar a profundidade, o problema piora de
forma **exponencial**: sem atalho, o gradiente é multiplicado por um fator
menor que 1 em cada camada no caminho de volta, e com 20 camadas a primeira
camada recebe ~10⁻¹⁶, ou seja, não aprende nada. Com atalho, a derivada de
`x + f(x)` em relação a `x` é `1 + f'(x)`: sempre existe o termo 1, que
leva o gradiente intacto até as primeiras camadas. É por isso que o GPT-2,
com 12 blocos (24 sub-camadas com atalho), consegue treinar, e por isso
tudo no bloco precisa manter `emb_dim` dimensões (a soma exige o mesmo
shape).

### Experimento 2 - Número de Transformer Blocks (`n_layers`)

| n_layers | Parâmetros | Nos blocos | Memória (MB) | Forward (ms) |
|---------:|-----------:|-----------:|-------------:|-------------:|
| 1 | 13.080.320 | 197.888 | 49,90 | 23,6 |
| 2 | 13.278.208 | 395.776 | 50,65 | 21,7 |
| 4 | 13.673.984 | 791.552 | 52,16 | 26,8 |
| 8 | 14.465.536 | 1.583.104 | 55,18 | 35,7 |
| 12 | 15.257.088 | 2.374.656 | 58,20 | 52,1 |

**Análise:** cada bloco acrescenta exatamente 197.888 parâmetros, então o
crescimento é **linear** em `n_layers`. Mas, no modelo MINI, passar de 1
para 12 blocos aumenta o total em só 17%: a maior parte dos parâmetros está
no vocabulário (ver Experimento 4). O tempo de forward cresce mais que os
parâmetros (23,6 → 52,1 ms, ≈ 2,2×), porque os blocos rodam **em
sequência**: cada um precisa da saída do anterior, então a profundidade não
pode ser paralelizada. Mais blocos permitem mais etapas de refinamento da
representação, mas só um modelo treinado mostra se isso compensa (Sprint 5).

### Experimento 3 - Número de heads (`emb_dim` = 128 fixo)

| n_heads | head_dim | Parâmetros | Forward (ms) |
|--------:|---------:|-----------:|-------------:|
| 1 | 128 | 13.673.984 | 25,5 |
| 2 | 64 | 13.673.984 | 25,9 |
| 4 | 32 | 13.673.984 | 29,9 |
| 8 | 16 | 13.673.984 | 36,1 |
| 16 | 8 | 13.673.984 | 32,4 |

**Análise:** o número de cabeças **não muda nenhum parâmetro**: as
projeções Q, K, V e `out_proj` continuam `emb_dim × emb_dim`; só a forma
como a dimensão é dividida muda (`head_dim = emb_dim / n_heads`). O custo
quase não muda (a variação é ruído de medição mais o custo de mais
matrizes de atenção pequenas). O que muda é a **capacidade de
representação**: com 16 cabeças o modelo pode aprender 16 padrões de
relação entre tokens diferentes, mas cada um num espaço de só 8 dimensões.
Por isso `emb_dim` precisa ser divisível por `n_heads` e, na prática,
`head_dim` fica em 64 (GPT-2 small: 768 / 12).

### Experimento 4 - Dimensão dos embeddings (`emb_dim`)

| emb_dim | Parâmetros | Nos blocos | Token emb + saída | Memória (MB) | Forward (ms) |
|--------:|-----------:|-----------:|------------------:|-------------:|-------------:|
| 64 | 6.640.384 | 199.168 | 96,9% | 25,3 | 23,7 |
| 128 | 13.673.984 | 791.552 | 94,1% | 52,2 | 26,4 |
| 256 | 28.920.832 | 3.155.968 | 89,0% | 110,3 | 53,9 |
| 512 | 64.133.120 | 12.603.392 | 80,2% | 244,7 | 120,4 |

**Análise:** é o parâmetro com maior impacto. Os blocos crescem com
**emb_dim²** (dobrar 128 → 256 multiplica os parâmetros dos blocos por 4,0),
enquanto o token embedding e a camada de saída crescem com
`vocab_size × emb_dim` (linear). Em modelos pequenos, quase tudo está no
vocabulário: com `emb_dim = 64`, 96,9% dos parâmetros são as duas matrizes
de 50.257 linhas, e só 3% fazem o "processamento" de verdade. Isso explica
por que o weight tying é atraente em modelos pequenos (elimina metade
desses parâmetros) e por que, à medida que o modelo cresce, a fração dos
blocos aumenta (no GPT-2 small, 52%). O tempo de forward cresce mais rápido
que `emb_dim` a partir de 256 (de 256 para 512, ≈ 2,2×), puxado pela camada
de saída (linear em `emb_dim`) e pelos blocos (quadráticos em `emb_dim`).

### Experimento 5 - Dimensão da FeedForward (`ff_hidden_dim`)

| ff_hidden_dim | FeedForward por bloco | Atenção por bloco | Parâmetros | Forward (ms) |
|---------------|----------------------:|------------------:|-----------:|-------------:|
| 1× = 128 | 33.024 | 65.664 | 13.279.232 | 24,1 |
| 2× = 256 | 65.920 | 65.664 | 13.410.816 | 24,4 |
| 4× = 512 (padrão) | 131.712 | 65.664 | 13.673.984 | 25,3 |
| 8× = 1024 | 263.296 | 65.664 | 14.200.320 | 29,8 |

**Análise:** os parâmetros da FeedForward crescem linearmente com a
dimensão interna (`2 × emb_dim × hidden`). Com o fator 4 do GPT-2, a
FeedForward tem o **dobro** de parâmetros da atenção (o mesmo resultado do
Exercício 4.1 no modelo grande: 4.722.432 contra 2.360.064). Com fator 2,
as duas ficam praticamente iguais. A expansão dá espaço para a FeedForward
combinar as features de cada token de forma não linear; a camada que
contrai de volta para `emb_dim` é obrigatória para somar o atalho.

### Experimento 6 - Dropout

**Configuração:** prompt `"Every effort moves you"`, dois forwards
seguidos com a mesma entrada, em modo de treino (`model.train()`) e de
avaliação (`model.eval()`).

| drop_rate | Zeros após `drop_emb` | Diferença média dos logits (treino) | Diferença (eval) |
|----------:|----------------------:|-----------------------------------:|-----------------:|
| 0,0 | 0,0% | 0,0000 | 0,0 |
| 0,1 | 9,0% | 0,2219 | 0,0 |
| 0,3 | 28,7% | 0,4251 | 0,0 |
| 0,5 | 51,6% | 0,5502 | 0,0 |

**Análise:** a fração de valores zerados acompanha a taxa configurada
(9,0% para 0,1, 51,6% para 0,5; a diferença é estatística, com só
4 × 128 valores). Em modo de treino, a mesma entrada produz logits
diferentes a cada chamada, tanto mais quanto maior a taxa: é o "ruído" que
impede o modelo de depender sempre das mesmas conexões (regularização
contra overfitting). Em `eval()` a diferença é exatamente 0: por isso a
geração chama `model.eval()` antes. O dropout não tem parâmetros. O
Exercício 4.3 está implementado: as chaves `drop_rate_emb`,
`drop_rate_attn` e `drop_rate_shortcut` controlam separadamente os três
lugares onde o dropout aparece (testado com 0,1 / 0,2 / 0,3).

### Experimento 7 - Tamanho do contexto

**Configuração:** MINI com `context_length = 512`, lote de 1 sequência.

| Tokens na entrada | Forward (ms) | ms por token | Elementos das matrizes de atenção |
|------------------:|-------------:|-------------:|----------------------------------:|
| 32 | 10,6 | 0,33 | 16.384 |
| 64 | 10,1 | 0,16 | 65.536 |
| 128 | 18,3 | 0,14 | 262.144 |
| 256 | 28,0 | 0,11 | 1.048.576 |
| 512 | 55,1 | 0,11 | 4.194.304 |

| context_length | Positional embedding | Buffer da máscara causal | Parâmetros |
|---------------:|---------------------:|-------------------------:|-----------:|
| 128 | 16.384 | 16.384 | 13.673.984 |
| 256 | 32.768 | 65.536 | 13.690.368 |
| 512 | 65.536 | 262.144 | 13.723.136 |
| 1.024 | 131.072 | 1.048.576 | 13.788.672 |

**Análise:** o número de elementos das matrizes de atenção cresce com o
**quadrado** do contexto (16× ao passar de 128 para 512), mas o tempo
medido cresceu só ~3×, praticamente linear. O motivo é que, neste modelo,
o custo por token é dominado pelas camadas lineares, em especial a camada
de saída: cada token faz `128 × 50.257 ≈ 6,4 M` multiplicações só nela,
contra `2 × n × emb_dim × n_layers ≈ 0,5 M` da atenção com 512 tokens. O
termo quadrático só passa a dominar com contextos bem maiores ou modelos
com `emb_dim` maior. Aumentar o **limite** (`context_length`) custa pouco
em parâmetros (só o positional embedding cresce, linear), mas a máscara
causal guardada como buffer cresce com o quadrado (1.048.576 valores para
1.024 posições). Por fim, o limite é rígido: uma sequência maior que
`context_length` gera erro, e é por isso que a geração corta o contexto.

### Experimento 8 - Inferência e estratégias de geração

**Configuração:** MINI não treinado, prompt `"Hello, I am"` →
`[15496, 11, 314, 716]`.

Uma iteração: logits `(1, 4, 50257)` → última posição `(1, 50257)` →
softmax (soma = 1,0). O token mais provável tem probabilidade 0,000151,
apenas 7,5× a de um chute uniforme (1/50.257 = 0,000020). A decodificação
gulosa gera `"Hello, I am interns safeguards neur nuisancebred PJ"` (10
tokens = 4 do prompt + 6 gerados).

Distribuição do próximo token:

| Temperatura | top-k | Prob. máxima | Entropia | Tokens efetivos (e^H) |
|------------:|------:|-------------:|---------:|----------------------:|
| 0,1 | - | 0,1343 | 3,83 | 46 |
| 0,5 | - | 0,0008 | 10,16 | 25.913 |
| 1,0 | - | 0,0002 | 10,66 | 42.513 |
| 2,0 | - | 0,0001 | 10,78 | 48.193 |
| 1,0 | 50 | 0,0265 | 3,90 | 50 |
| 1,0 | 5 | 0,2087 | 1,61 | 5 |
| 2,0 | 5 | 0,2044 | 1,61 | 5 |

Entropia máxima (uniforme) = ln(50.257) = 10,83.

Diversidade em 10 gerações de 6 tokens: a gulosa produz sempre a mesma
sequência (1 distinta, 6 tokens únicos); com sorteio, as 10 sequências são
distintas (60 tokens únicos com T = 0,5, 1,0 e 2,0; 51 com top-k = 5).

Custo da geração (sem KV cache): 10 tokens em 111 ms, 80 tokens em 946 ms,
cerca de 11 ms por token.

**Análise:**

- O modelo não treinado tem uma distribuição **quase uniforme** (entropia
  10,66 contra o máximo de 10,83), e por isso o texto não faz sentido,
  qualquer que seja a estratégia. Só depois do treino (Sprint 5) os logits
  passam a separar tokens plausíveis dos absurdos.
- A **temperatura** controla a concentração: com T = 0,1 os logits são
  multiplicados por 10, e o favorito sobe de 0,015% para 13,4%, restando
  ~46 candidatos efetivos; com T = 2 a distribuição fica ainda mais perto
  da uniforme. No limite T → 0 recuperamos a gulosa.
- O **top-k** corta a cauda: com k = 5 restam exatamente 5 candidatos
  efetivos (entropia = ln 5 = 1,61). Como os 5 maiores logits do modelo não
  treinado são quase iguais, a temperatura praticamente não muda nada
  dentro deles (T = 1 e T = 2 dão a mesma entropia). Num modelo treinado,
  temperatura moderada + top-k dá variedade sem permitir tokens absurdos.
- Gulosa = determinística (mesma entrada → mesma saída); sorteio =
  variado, e reprodutível só com `torch.manual_seed`.
- O custo por token ficou estável (~11 ms) porque, com contextos curtos,
  o forward é dominado pelo custo fixo da camada de saída (Experimento 7).
  Mesmo assim, cada token gerado exige um **forward completo** sobre todo o
  contexto: sem KV cache, os vetores K e V dos tokens anteriores são
  recalculados a cada passo, e o custo total cresce com o comprimento do
  texto.

### Experimento 9 - GPT-2 small real, weight tying e Exercício 4.2

O `GPTModel(GPT_CONFIG_124M)` instanciado tem **163.009.536** parâmetros
(124.412.160 descontando a camada de saída), **621,83 MB** em float32, e
faz um forward de 64 tokens em 131 ms na CPU. Com `weight_tying=True`, o
mesmo modelo tem **124.412.160** parâmetros (474,59 MB): a camada de saída
passa a ser literalmente o mesmo tensor do token embedding.

| Modelo | emb_dim | n_layers | n_heads | Parâmetros | Com weight tying | Memória (MB) |
|--------|--------:|---------:|--------:|-----------:|-----------------:|-------------:|
| GPT-2 small | 768 | 12 | 12 | 163.009.536 | 124.412.160 | 621,83 |
| GPT-2 medium | 1.024 | 24 | 16 | 406.212.608 | 354.749.440 | 1.549,58 |
| GPT-2 large | 1.280 | 36 | 20 | 838.220.800 | 773.891.840 | 3.197,56 |
| GPT-2 XL | 1.600 | 48 | 25 | 1.637.792.000 | 1.557.380.800 | 6.247,68 |

**Análise:** os valores batem com o Exercício 4.2 do livro. Os modelos
maiores mudam só `emb_dim`, `n_layers` e `n_heads`; o código é o mesmo. Do
small para o XL, os parâmetros crescem 10×, a maior parte nos blocos
(`n_layers × ~12 × emb_dim²`). Só os pesos do XL já ocupam 6,2 GB, sem
contar gradientes e estado do otimizador para treinar, o que justifica o
uso da configuração MINI no projeto.

### Experimento 10 - Pipeline completo das Sprints 2 a 4

`the-verdict.txt` → tokenizador BPE → `create_dataloader_v1` (Sprint 2,
lote 8 × 64) → `GPTModel` MINI → logits `(8, 64, 50257)`. Como prévia do
Capítulo 5, calculamos a entropia cruzada entre os logits e os alvos
(entrada deslocada em 1 token): **10,989**, muito perto de
ln(50.257) = 10,825, o valor de um modelo que chuta uniformemente. Esse é
o ponto de partida do treinamento: na Sprint 5, a perda precisa cair
abaixo desse patamar para o texto gerado começar a fazer sentido.

## 6. Conclusões

- A arquitetura GPT é uma composição simples e repetitiva: embeddings, uma
  pilha de blocos idênticos e uma camada de saída. O tamanho vem dos
  parâmetros, não do código: o mesmo `GPTModel` gera do MINI (13,7 M) ao
  GPT-2 XL (1,6 B) trocando só o `cfg`.
- Dentro do modelo tudo mantém o shape `(batch, num_tokens, emb_dim)`; isso
  é condição para os atalhos (somas) e para empilhar blocos.
- As conexões de atalho são o que torna redes profundas treináveis: sem
  elas, o gradiente da primeira camada cai exponencialmente com a
  profundidade (10⁻¹⁶ com 20 camadas).
- `emb_dim` é o hiperparâmetro de maior impacto no custo (quadrático nos
  blocos); `n_heads` não muda parâmetros; `n_layers` e `ff_hidden_dim`
  crescem linearmente. Em modelos pequenos, o vocabulário domina os
  parâmetros.
- A geração autoregressiva transforma um modelo que só prevê "o próximo
  token" em um gerador de texto; temperatura e top-k controlam o equilíbrio
  entre previsibilidade e variedade.
- Sem treino, o modelo está no nível do chute uniforme (perda ≈ ln V). O
  próximo passo (Sprint 5) é o laço de treinamento com essa mesma perda.

## 7. O que a Sprint 5 vai usar daqui

- `GPTModel` e `GPT_CONFIG_MINI` como modelo a ser treinado;
- os logits `(batch, num_tokens, vocab_size)` e a entropia cruzada do
  Experimento 10 como função de perda;
- `generate` (com temperatura e top-k) para avaliar o texto gerado ao longo
  do treino;
- `weight_tying` e `qkv_bias`, necessários para carregar os pesos oficiais
  do GPT-2.
