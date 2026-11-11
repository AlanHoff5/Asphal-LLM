# Sprint 5 - Treinamento do modelo GPT

## 1. Objetivo

A Sprint 4 terminou com um GPT completo, mas com pesos aleatórios: ele gera
texto sem sentido e erra no nível do chute uniforme. A Sprint 5 implementa o
**pré-treinamento** (Capítulo 5 do livro): medir o erro do modelo, treinar
os pesos para diminuir esse erro, controlar a geração de texto e salvar o
resultado.

```text
Texto (The Verdict) → 90% treino / 10% validação → DataLoaders (Sprint 2)
  ↓
Laço de treino, para cada lote:
  zero_grad → GPTModel (Sprint 4) → logits → cross-entropy → backward → AdamW.step
  ↓  a cada 5 passos: perda de treino e validação | a cada época: amostra de texto
Modelo treinado → geração (gulosa, temperatura, top-k) → salvar / carregar (.pth)
```

O código reutilizável fica em `src/training/` (cumulativo, como `src/model/`):

| Arquivo | Conteúdo |
|---------|----------|
| `dispositivo.py` | escolhe GPU (CUDA) ou CPU e diagnostica a instalação do PyTorch |
| `dados.py` | divisão 90/10 e DataLoaders de treino e validação (reaproveita `create_dataloader_v1`) |
| `losses.py` | `calc_loss_batch`, `calc_loss_loader`, perplexity |
| `treino.py` | `train_model_simple` (seção 5.2) e `train_model` (apêndice D: warmup, cosine decay, gradient clipping) |
| `graficos.py` | gráfico de perdas (figura 5.12) e comparação entre experimentos |
| `checkpoint.py` | salvar e carregar pesos, otimizador e configuração |

**Ambiente dos resultados abaixo:** GPU NVIDIA GeForce GTX 1650 (4 GB),
PyTorch 2.11.0+cu128, semente 123, texto *The Verdict* (20.479 caracteres,
5.145 tokens BPE).

## 2. Entregável 2 - Avaliação do modelo

Script: `experiments/avaliacao_modelo.py` → `resultados/avaliacao_inicial_124m.json`.

### 2.1 Cross-entropy passo a passo (seção 5.1.2)

Com o GPT-2 124M sem treino (contexto 256), as entradas *"every effort
moves"* e *"I really like"* e os alvos deslocados uma posição:

| Passo | Resultado |
|-------|-----------|
| logits | `(2, 3, 50257)` |
| softmax → probabilidade do token certo, frase 1 | 7,454e-05 · 3,106e-05 · 1,156e-05 |
| log das 6 probabilidades | -9,50 · -10,38 · -11,37 · -11,48 · -9,78 · -12,26 |
| média dos logs | -10,7940 |
| troca de sinal = cross-entropy | **10,7940** |
| `torch.nn.functional.cross_entropy` (logits `(6, 50257)`, alvos `(6,)`) | 10,7940 |

As probabilidades são **idênticas às do livro**, o que confirma que o
`GPTModel` do projeto reproduz a inicialização do livro. Para o modelo, o
token certo tem probabilidade da ordem de 1/50.000: ele está chutando.

### 2.2 Perplexity

Perplexity = e^perda. Ela traduz a perda no número de tokens entre os quais
o modelo está "em dúvida" a cada passo:

| Situação | Perda | Perplexity |
|----------|------:|-----------:|
| exemplo do livro, sem treino | 10,794 | 48.725,8 |
| chute uniforme, ln(50.257) | 10,825 | 50.257 |
| treino do livro, validação final | 6,452 | 634 |
| treino do livro, treino final | 0,420 | 1,5 |

### 2.3 Treino x validação e estado inicial (seção 5.1.3)

| | Lotes | Tokens | Perda inicial | Perplexity |
|---|---:|---:|---:|---:|
| Treino (90%) | 9 | 4.608 | 10,9876 | 59.135 |
| Validação (10%) | 1 | 512 | 10,9811 | 58.754 |

Números iguais aos do livro (10,98758 e 10,98110). As duas perdas ficam
perto de ln(50.257): sem treino, o modelo não tem preferência por nenhum
token. Treino e validação quase iguais indicam que ainda não há
overfitting. O texto gerado também é o do livro: *"Every effort moves you
rentingetic wasnم refres RexMeCHicular stren"*.

## 3. Entregável 3 - Implementação do treinamento

Script: `experiments/treino_modelo.py` → `resultados/treino_124m.json`,
gráfico `resultados/treino_124m.png` (gerado localmente; os `.png` ficam
fora do git) e pesos em `models/ch05/treino_124m.pth`.

### 3.1 O laço de treino

`train_model_simple` segue a seção 5.2. Para cada época e para cada lote:

1. `optimizer.zero_grad()`: apaga os gradientes do lote anterior (o PyTorch acumula);
2. `calc_loss_batch(...)`: forward e cross-entropy;
3. `loss.backward()`: retropropagação, um gradiente para cada um dos 162 M de pesos;
4. `optimizer.step()`: o AdamW atualiza os pesos.

A cada `eval_freq = 5` passos, `evaluate_model` mede a perda média de
`eval_iter = 5` lotes de treino e de validação em `model.eval()` (sem
dropout, sem gradientes) e volta para `model.train()`. Ao fim de cada
época, uma amostra gulosa de 50 tokens mostra a evolução do texto.

### 3.2 Parâmetros utilizados

| Parâmetro | Valor |
|-----------|-------|
| Modelo | GPT-2 124M com `context_length = 256` (162.419.712 parâmetros) |
| Épocas | 10 (9 lotes por época, 90 passos) |
| Tamanho do lote | 2 sequências × 256 tokens |
| Otimizador | AdamW, `lr = 0,0004`, `weight_decay = 0,1`, betas (0,9; 0,999) |
| Dropout | 0,1 |
| Avaliação | `eval_freq = 5`, `eval_iter = 5` |
| Semente | 123 |

### 3.3 Resultado

| | Antes | Depois | Livro |
|---|---:|---:|---:|
| Perda de treino | 10,988 | **0,420** | 0,391 |
| Perda de validação | 10,981 | **6,452** | 6,452 |
| Perplexity de validação | 58.754 | 634 | - |

- **Custo:** 63 s na GTX 1650 (cerca de 4,5 s por época, incluindo avaliações
  e amostras), 3,07 GB de memória de GPU (de 4 GB).
- **O modelo aprende:** a perda de treino cai de 10,99 para 0,42. As
  amostras mostram a progressão: só pontuação ("Every effort moves you.")
  → palavras frequentes ("the", "I", "to the picture") → frases do conto
  ("He laughed again, and threw back his head to look up at the sketch of
  the donkey").
- **Mas decora:** a validação só melhora até a época 5 (mínimo **6,087** no
  passo 45) e depois volta a subir, enquanto o treino continua caindo. É o
  overfitting que o livro descreve: 4.608 tokens de treino para 162 M de
  parâmetros.
- **Um pico no passo 20:** a perda saltou de cerca de 6 para 15 e voltou a
  5,4 no passo 25. Isso não aparece no livro e é analisado na seção 4.3.

## 4. Entregável 4 - Experimentos de treinamento

Script: `experiments/experimentos_treino.py` → `resultados/expNN_*.json`,
`resultados/experimentos_resumo.md` e os gráficos `comparacao_validacao.png`
e `comparacao_treino.png`. Cada experimento muda **um** parâmetro em
relação ao baseline (o treino do livro). A fila inteira leva 12,8 min.

| Experimento | O que mudou | Treino final | Val. final | Menor val. (época) | Gap val-treino | Tempo | GPU |
|-------------|-------------|---:|---:|---:|---:|---:|---:|
| exp01_baseline | (livro) | 0,420 | 6,452 | **6,087** (5,1) | 6,03 | 67 s | 3,07 GB |
| exp02_lr_alta | lr 0,001 | 5,488 | 6,819 | 6,763 (2,3) | 1,33 | 65 s | 3,07 GB |
| exp03_lr_baixa | lr 0,0001 | 2,900 | **6,211** | 6,189 (9,6) | 3,31 | 86 s | 3,07 GB |
| exp04_30_epocas | 30 épocas | **0,006** | 7,297 | 6,087 (5,1) | 7,29 | 262 s | 3,07 GB |
| exp05_sem_regularizacao | dropout 0, wd 0 | 0,661 | 6,352 | 6,132 (6,2) | 5,69 | 69 s | 3,07 GB |
| exp06_warmup_cosine_clip | lr 0,001 + warmup 20% + cosine + clip 1,0 | 1,217 | 6,230 | 6,180 (6,2) | 5,01 | 64 s | 3,07 GB |
| exp07_modelo_mini | 13,7 M parâmetros, contexto 128 | 5,411 | 6,445 | 6,445 (10) | 1,03 | 19 s | 0,34 GB |
| exp08_modelo_small | 30,5 M parâmetros, contexto 256 | 4,492 | 6,229 | 6,229 (10) | 1,74 | 19 s | 0,80 GB |

### 4.1 Taxa de aprendizado (exp01, exp02, exp03)

- **lr 0,001 (alta demais):** o modelo não converge. A perda de treino fica
  presa em torno de 5,5 a 6, com um salto no passo 45 (validação 10,77), e
  o texto degenera em repetições ("the, the, the", ",,,,,,", "deadline
  deadline deadline"). Passos grandes demais fazem os pesos "pularem" o
  mínimo, em vez de descer até ele.
- **lr 0,0001 (baixa):** a descida é lenta e suave (sem picos), e depois de
  10 épocas o treino ainda está em 2,90. Como o modelo decorou menos, a
  validação final (6,211) é **melhor** que a do baseline (6,452), e ainda
  estava caindo.
- **lr 0,0004 (livro):** desce mais rápido e atinge a menor validação de
  todas (6,087), mas na época 5. A partir daí, cada época a mais só aumenta
  a memorização.

### 4.2 Número de épocas e overfitting (exp01, exp04)

Com 30 épocas, a perda de treino chega a **0,006** (perplexity 1,0: o
modelo prevê o conto com certeza absoluta) e a validação sobe para **7,297**,
pior que em qualquer momento do baseline. As épocas 6 a 30 só pioraram a
generalização. Da época 11 em diante, todas as amostras são a mesma frase
copiada do conto: *"Yes--quite insensible to the irony. She wanted him
vindicated--and by me!"*. A menor validação continua na época 5. Com este
volume de dados, treinar mais não ajuda; o certo seria parar perto do
mínimo de validação (*early stopping*).

### 4.3 Regularização, warmup e o pico do passo 20 (exp05, exp06)

- **Sem dropout e sem weight decay (exp05):** o gap final (5,69) ficou
  parecido com o do baseline (6,03). Em 10 épocas com 5 mil tokens, o
  efeito da regularização é pequeno, e a comparação fica confundida pelo
  pico do passo 20, que **não** aconteceu neste experimento.
- **Warmup + cosine + clipping (exp06):** com o **mesmo lr de pico 0,001**
  do exp02, o modelo agora aprende: validação 6,230 contra 6,819 do exp02,
  e texto com frases do conto em vez de repetições. O warmup (18 passos,
  lr de 0,0001 até 0,001) e a descida em cosseno até 0,00001 tornam um lr
  alto utilizável. No fim do treino, o lr pequeno também reduz a
  memorização (treino 1,217 contra 0,420 no baseline).
- **O pico do passo 20:** a perda foi a cerca de 15 no passo 20 do exp01, do
  exp04 e do exp06, que usam a mesma ordem de lotes e dropout 0,1, e não
  apareceu no exp03 (lr baixo) nem no exp05 (sem dropout). O gradient
  clipping do exp06 **não** evitou o pico. Uma explicação provável: o Adam
  divide cada atualização pela média móvel do gradiente ao quadrado, então
  limitar a norma do gradiente não limita o tamanho do passo. O salto vem
  da combinação de um lote específico, da máscara de dropout e de um lr
  alto naquele momento. O livro, treinando na CPU, não mostra esse pico;
  pequenas diferenças numéricas da GPU mudam a trajetória.

### 4.4 Tamanho do modelo e custo (exp01, exp07, exp08)

| Modelo | Parâmetros | Val. final | Gap | Tempo | Memória |
|--------|---:|---:|---:|---:|---:|
| mini | 13,7 M | 6,445 | 1,03 | 19 s | 0,34 GB |
| small | 30,5 M | 6,229 | 1,74 | 19 s | 0,80 GB |
| 124M (livro) | 162,4 M | 6,452 (mínimo 6,087) | 6,03 | 67 s | 3,07 GB |

O modelo de 124M é 12 vezes maior que o *mini* e 3,5 vezes mais lento, mas
termina com a mesma validação. A diferença está no **gap**: o modelo
grande decora o treino (gap 6,03), enquanto os pequenos ainda estavam
aprendendo padrões gerais quando o treino acabou (validação ainda caindo).
Com 4.608 tokens de treino, capacidade extra vira memorização. Nos
modelos pequenos, o tempo é dominado pelo custo fixo de avaliação e
geração das amostras, por isso o *mini* e o *small* levaram o mesmo tempo.

### 4.5 Conclusão dos experimentos

Com o texto do livro, **nenhuma configuração passou de uma perda de
validação de cerca de 6,1** (perplexity cerca de 450). O limite é a
quantidade de dados, não o modelo nem o otimizador. O que os experimentos
controlam é **como** se chega lá: lr alto demais não converge, lr baixo ou
modelos menores memorizam menos, mais épocas só aumentam o overfitting, e
warmup com cosine torna um lr alto utilizável.

## 5. Entregável 5 - Estratégias de geração

Script: `experiments/estrategias_geracao.py` → `resultados/estrategias_geracao.md`
(tabela e todos os textos) e `.json`.

### 5.1 Exemplo do livro: 9 palavras, 1000 sorteios (seção 5.3)

| Palavra | Logit | T = 1 | T = 0,1 | T = 5 | top-k = 3 |
|---------|------:|------:|--------:|------:|----------:|
| closer | 4,51 | 73 | 0 | 165 | 73 |
| every | 0,89 | 0 | 0 | 75 | 0 |
| effort | -1,90 | 0 | 0 | 42 | 0 |
| forward | 6,75 | **582** | **985** | 239 | 583 |
| inches | 1,63 | 2 | 0 | 71 | 0 |
| moves | -1,62 | 0 | 0 | 46 | 0 |
| pizza | -1,89 | 0 | 0 | 32 | 0 |
| toward | 6,28 | 343 | 15 | 227 | 344 |
| you | 1,79 | 0 | 0 | 103 | 0 |

A coluna T = 1 reproduz o livro exatamente. A gulosa escolheria sempre
*forward*. T = 0,1 fica quase igual à gulosa (985 de 1000). T = 5 espalha
os sorteios a ponto de *pizza*, que não faz sentido na frase, sair 32
vezes. O top-k = 3 só deixa sortear *closer*, *forward* e *toward*.

### 5.2 Mesmas estratégias no GPT treinado e no GPT sem treino

Contexto *"Every effort moves you"*, 30 tokens novos, 3 sementes por
estratégia. As métricas usam só os tokens gerados: *distinct-1/2* é a
fração de tokens e pares únicos (diversidade), *repetição* é a fração de
trigramas repetidos, e *cópia do treino* é a fração de sequências de 8
tokens idênticas ao texto de treino.

| Modelo treinado | distinct-1 | distinct-2 | repetição | cópia do treino | amostras distintas |
|-----------------|---:|---:|---:|---:|---:|
| gulosa | 0,60 | 0,72 | 0,21 | 0,00 | 1/3 |
| amostragem T = 1,0 | 0,89 | 0,98 | 0,00 | 0,01 | 3/3 |
| T = 0,5 | 0,68 | 0,78 | 0,17 | 0,19 | 2/3 |
| T = 1,5 | 1,00 | 1,00 | 0,00 | 0,00 | 3/3 |
| top-k = 3 | 0,72 | 0,80 | 0,15 | 0,26 | 3/3 |
| top-k = 40 | 0,73 | 0,85 | 0,08 | 0,28 | 3/3 |
| top-k = 25, T = 1,4 (livro) | 0,91 | 0,99 | 0,00 | 0,07 | 3/3 |

- **Gulosa:** as 3 sementes dão o mesmo texto (é determinística), com a
  maior repetição. O texto é uma colagem de trechos curtos do conto (*"I
  glanced after him… He laughed again, and"*), por isso a cópia de 8
  tokens seguidos fica em 0.
- **Temperatura baixa e top-k pequeno** mantêm o texto perto do que o modelo
  viu: as maiores taxas de cópia literal (0,19 a 0,28) e textos coerentes,
  mas pouco originais.
- **T = 1,5 sem top-k:** diversidade máxima, mas o texto perde o sentido e
  vira palavras soltas do vocabulário inteiro (*"enzymerow Kerry me
  UEEgypt…"*), porque tokens de baixa probabilidade passam a ser sorteados.
- **top-k = 25 com T = 1,4 (configuração do livro):** o melhor equilíbrio.
  É bem variada (distinct-2 0,99, sem repetição), mas fica restrita a
  palavras plausíveis e pouco copia o treino (*"Stroud tears I felt able to
  face the fact with such--had not when she began to stammer something
  about her again"*).

**Estratégia x treinamento:** no modelo **sem treino**, todas as 7
estratégias dão diversidade de cerca de 1,00 e cópia 0, e todos os textos
são sequências aleatórias do vocabulário (*"Pleasant ampERO superfraise
adequate bear…"*). A estratégia de geração só escolhe **entre** o que o
modelo considera provável: ela controla diversidade e repetição, mas não
cria conhecimento. A coerência vem inteiramente do treino dos pesos.

## 6. Entregável 6 - Salvamento e carregamento dos pesos

Script: `experiments/salvar_carregar.py` → `resultados/salvar_carregar.json`.
Funções em `src/training/checkpoint.py`.

O arquivo `models/ch05/treino_124m.pth` (1.862 MB) guarda:

| Chave | Conteúdo |
|-------|----------|
| `model_state_dict` | os pesos (`model.state_dict()`) |
| `optimizer_state_dict` | o estado do AdamW (duas médias móveis por peso) |
| `cfg` | a configuração do GPT, necessária para recriar o modelo antes de carregar os pesos |
| `parametros` | os parâmetros do treino (épocas, lote, lr, semente, versão do PyTorch) |

Para carregar, `carregar_modelo` lê o arquivo na CPU
(`torch.load(..., map_location="cpu", weights_only=True)`), cria
`GPTModel(cfg)`, aplica `load_state_dict`, move para a GPU e chama
`model.eval()`. A leitura passa pela CPU para não duplicar o otimizador na
GPU de 4 GB.

**Demonstração de que o modelo carregado é o mesmo do fim do treino:**

| Verificação | Resultado |
|-------------|-----------|
| Perda de validação registrada no fim do treino | 6,452223 |
| Perda de validação recalculada após carregar | 6,452223 (diferença 0) |
| Texto gerado (gulosa, 50 tokens) igual à última amostra do treino | **sim**, idêntico |
| Tamanho só com os pesos (`model.pth` do livro) | 623 MB |
| Tamanho com pesos + otimizador | 1.862 MB |
| Continuar o treino por 1 época a partir do arquivo | perda de treino 0,420 → 0,234 |

O otimizador ocupa cerca de 2/3 do arquivo e só é necessário para
**continuar** o treino. Sem ele, o AdamW recomeçaria sem histórico. Com
ele, o treino continua de onde parou, sem voltar para cerca de 11. A época
extra reduziu o treino e piorou a validação (6,46 → 6,56): mais
overfitting, como previsto na seção 4.2.

**Reprodutibilidade:** os `.pth` não vão para o git (o `.gitignore` ignora
`models/` e `*.pth`; o GitHub recusa arquivos acima de 100 MB). Cada JSON
em `resultados/` guarda a configuração completa usada, e os pesos são
recriados rodando o mesmo script com a mesma semente (123).

## 7. Limitações

- **Dados:** 5.145 tokens de um único conto. Isso é suficiente para
  observar o treinamento, mas não para o modelo generalizar. A validação
  (512 tokens, 1 lote) é o trecho final do conto e é estatisticamente
  pequena.
- **Determinismo na GPU:** com a mesma semente, a GPU produz trajetórias um
  pouco diferentes das do livro (CPU). O pico do passo 20 é um exemplo;
  experimentos com diferenças pequenas (como o exp05) devem ser
  interpretados com cautela.
- **Memória:** o GPT-2 124M com AdamW usa 3,07 dos 4 GB da GTX 1650 com
  lote 2. Lotes maiores ou contexto 1024 não cabem sem técnicas extras
  (precisão mista, acumulação de gradientes).
- **Métricas de geração:** distinct-n, repetição e cópia medem diversidade e
  memorização, não qualidade. Coerência foi avaliada lendo os textos.
- **Sem pesos oficiais:** a seção 5.5 (carregar o GPT-2 da OpenAI) não fez
  parte do escopo da Sprint 5.

## 8. Conclusões

- A cross-entropy mede quanta probabilidade o modelo dá ao token certo; a
  perplexity traduz isso em "entre quantos tokens ele está em dúvida". Sem
  treino: perda cerca de 11 e perplexity de cerca de 50 mil (chute
  uniforme).
- O laço de treino tem quatro passos (`zero_grad`, forward + perda,
  `backward`, `step`) e, com a configuração do livro, reproduz os números
  do livro em 1 minuto na GTX 1650.
- Com um texto pequeno, o modelo de 124M decora o treino (0,42) sem
  generalizar (6,45). A menor perda de validação está perto da época 5;
  treinar mais só aumenta o overfitting.
- Taxa de aprendizado é o hiperparâmetro mais sensível. Warmup com cosine
  decay torna utilizável uma taxa que, sozinha, impede a convergência.
- Estratégias de geração controlam o equilíbrio entre repetição e
  diversidade (top-k = 25 com T = 1,4 foi o melhor), mas a coerência vem
  dos pesos treinados.
- Salvar pesos, otimizador e configuração permite reproduzir exatamente o
  modelo e continuar o treino.

## 9. O que a Sprint 6 vai usar daqui

- `carregar_modelo` e o formato de checkpoint para partir de um modelo já
  treinado;
- `train_model_simple` e `calc_loss_loader` como base do laço de fine-tuning;
- `generate` com temperatura e top-k para avaliar as respostas;
- a conclusão de que treinar do zero com poucos dados só memoriza: o
  fine-tuning parte de pesos pré-treinados (GPT-2 da OpenAI, seção 5.5),
  o que exige `qkv_bias = True`, `context_length = 1024` e `weight_tying`.
