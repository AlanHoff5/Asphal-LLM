# Apêndice D - Incrementando o laço de treino

## 1. Objetivo

O Apêndice D acrescenta três técnicas ao laço de treino do Capítulo 5 para
deixar o treino mais estável: **warmup** da taxa de aprendizado, **cosine
decay** e **gradient clipping**. O laço modificado (`train_model`) e o
cronograma (`taxa_de_aprendizado`) já tinham sido implementados no Capítulo 5
e usados no `exp06`; este apêndice documenta cada técnica e **separa o efeito
de cada uma** num experimento com três treinos.

```text
train_model_simple (Cap. 5)
  + warmup:   lr sobe de initial_lr até peak_lr nos primeiros 20% dos passos
  + cosine:   depois desce em meia onda de cosseno até min_lr
  + clipping: depois do warmup, se ||g|| > 1.0, todos os gradientes × 1.0 / ||g||
= train_model (Apêndice D)
```

O código reutilizável fica em `src/training/`:

| Arquivo | Conteúdo |
|---------|----------|
| `treino.py` | `taxa_de_aprendizado` e `train_model` (já existiam, do Cap. 5) |
| `gradientes.py` | `find_highest_gradient` (do livro) e `norma_do_gradiente` (a norma usada pelo `clip_grad_norm_`) |

**Ambiente dos resultados abaixo:** CPU de 2 threads, GPT-2 124M com
contexto 256, *The Verdict* (9 lotes de treino por época), semente 123.
Script: `experiments/fluxo_apendiceD.py` → `resultados/apendiceD_resultados.json`.

## 2. Warmup e cosine decay (seções D.1 e D.2)

Com os valores da demonstração do livro (`initial_lr = 0,0001`,
`peak_lr = 0,01`, `min_lr = 0,1 · initial_lr`, 15 épocas de 9 lotes):

| Passo | Só warmup (D.1) | Warmup + cosine (D.2) | Momento |
|------:|----------------:|----------------------:|---------|
| 0 | 1,00e-04 | 1,00e-04 | início do warmup |
| 27 | 1,00e-02 | 1,00e-02 | pico |
| 81 | 1,00e-02 | 5,01e-03 | meio do cosine |
| 134 | 1,00e-02 | 1,21e-05 | último passo |

São 135 passos e `warmup_steps = int(0,2 · 135) = 27`, o mesmo número do
livro. No meio da descida a taxa vale exatamente a média entre o pico e o
mínimo, e no último passo fica a um passo de `min_lr`. Os testes conferem
cada uma dessas propriedades (subida em linha reta, descida que nunca sobe,
valor no meio do cosine).

## 3. Gradient clipping (seção D.3)

Um lote do GPT-2 124M sem treino, depois do `backward`:

| Medida | Antes | Depois de `clip_grad_norm_(max_norm=1.0)` |
|--------|------:|------------------------------------------:|
| Maior gradiente individual | 0,0552 | 0,0243 |
| Norma L2 de todos os gradientes | 2,2766 | 1,0000 |

Como a norma passa de 1,0, todos os gradientes são multiplicados por
1,0 / 2,2766 = 0,4393; o maior gradiente cai na mesma proporção. O livro
mostra 0,0446 → 0,0201, uma redução quase igual (fator 0,45), em outra
máquina. A direção do passo não muda: o teste
`test_clipping_limita_a_norma_e_mantem_a_direcao` confere que cada
gradiente foi multiplicado pelo mesmo fator.

No modelo mini (13,7 M parâmetros) a norma inicial é 0,90, menor que 1,0, e
o clipping não muda nada. Por isso, no mini, os treinos com e sem clipping
dão praticamente o mesmo resultado (perda de validação 6,277 nos dois).

## 4. Os três treinos (seção D.4)

Configuração do livro na seção D.4: lr de pico 0,001, `initial_lr` e
`min_lr` de 0,00001, 20% de warmup (27 passos), 15 épocas, `eval_iter = 1`,
AdamW com `weight_decay = 0,1`. Os três treinos usam a mesma semente e a
mesma ordem de lotes; só muda o que o apêndice acrescenta.

| Treino | Perda treino | Perda val. final | Melhor val. | Tempo (CPU) |
|--------|-------------:|-----------------:|------------:|------------:|
| `constante`: lr fixo em 0,001 | 0,875 | 7,133 | 6,593 (época 2,9) | 734 s |
| `warmup_cosine`: sem clipping | 0,900 | **6,368** | 6,238 (época 9,6) | 536 s |
| `warmup_cosine_clip`: o do livro | **0,032** | 6,938 | **6,201** (época 6,2) | 608 s |

### 4.1 O cronograma é o que mais ajuda

Com lr constante de 0,001, a validação chega ao melhor valor (6,593) já na
época 2,9 e depois só piora, até 7,133. O mesmo lr de pico com warmup e
cosine chega a 6,238, termina em 6,368 e a curva de validação fica estável
nas últimas épocas, quando a taxa já está pequena. É o mesmo padrão do
Capítulo 5 na GTX 1650 (10 épocas): `exp02` com lr 0,001 constante terminou
com validação 6,819, e o `exp06` com os extras terminou com 6,230.

### 4.2 O clipping acelera o treino, e com isso a memorização

O resultado menos óbvio está no terceiro treino. Até o passo 27 (fim do
warmup) ele é idêntico ao segundo, porque o clipping só entra depois do
warmup. A partir daí, a perda de treino com clipping cai muito mais rápido:

| Passo (época) | Treino sem clipping | Treino com clipping |
|---------------|--------------------:|--------------------:|
| 30 (3,4) | 5,31 | 4,62 |
| 45 (5,1) | 4,88 | 2,48 |
| 60 (6,8) | 4,81 | 0,82 |
| 130 (14,6) | 0,83 | 0,03 |

O modelo com clipping atinge a **melhor validação dos três** (6,201) mais
cedo, na época 6,2, mas depois decora o texto: a perda de treino vai a
0,03 e a amostra final é uma cópia literal de um trecho do conto ("Yes--quite
insensible to the irony..."). A validação sobe de volta até 6,938.

Uma explicação provável: o AdamW divide cada passo pela média móvel do
gradiente ao quadrado, com memória longa (β₂ = 0,999). Sem clipping, os
gradientes grandes do início inflam essa média, e os passos seguintes ficam
menores. Com clipping, todos os gradientes chegam com norma no máximo 1,0,
a média fica menor e os passos efetivos ficam **maiores**. Em 5 mil tokens
de treino, passos maiores significam decorar mais rápido. Com um corpus do
tamanho de um pré-treino de verdade, essa mesma rapidez é bem-vinda; aqui,
o que salvaria o resultado é parar o treino perto da época 6 (*early
stopping*).

### 4.3 Nenhum pico de perda na CPU

A maior perda registrada em cada treino é a do primeiro passo: não houve
picos. No Capítulo 5, na GTX 1650, a perda saltou para cerca de 15 no passo
20 do `exp01` e do `exp06`, mesmo com clipping. A diferença numérica entre
GPU e CPU muda a trajetória, e na CPU o salto não acontece, como também não
acontece no livro.

## 5. Comparação com o modelo mini

`--config mini` (13,7 M parâmetros, 18 lotes por época, mesmas 15 épocas):

| Treino | Perda treino | Perda val. final | Melhor val. |
|--------|-------------:|-----------------:|------------:|
| `constante` | 1,239 | 6,571 | 6,167 (época 9,5) |
| `warmup_cosine` | 4,426 | 6,277 | 6,238 (época 14,8) |
| `warmup_cosine_clip` | 4,427 | 6,277 | 6,238 (época 14,8) |

No mini, o clipping é inativo (seção 3) e os dois últimos treinos
coincidem. O lr constante decora mais (treino 1,239) e termina pior na
validação, o mesmo sentido do 124M.

## 6. Limitações

- O corpus tem 5.145 tokens. Qualquer configuração acaba decorando o texto,
  então a perda de validação mede mais "quando a memorização começa" do que
  a qualidade de um modelo de linguagem.
- Cada configuração rodou com uma semente só. As diferenças de 0,04 na
  melhor validação (6,201 contra 6,238) estão dentro do que outra semente
  poderia mudar; as diferenças de 0,4 a 0,7 na validação final, não.
- Os tempos são de uma CPU de 2 threads. Na GTX 1650, cada treino leva
  cerca de 1,5 min (o Cap. 5 mediu 67 s para 10 épocas).

## 7. Conclusões

1. **Warmup + cosine** é a parte que mais ajuda com lr alto: mesma taxa de
   pico, validação final 6,368 contra 7,133 do lr constante.
2. **Gradient clipping** com AdamW não serve só como "freio": ao uniformizar
   a escala dos gradientes, ele deixou o treino mais rápido. Isso deu a
   melhor validação de todas (6,201), mas também a pior memorização.
3. Para este corpus pequeno, a receita completa do livro pede *early
   stopping* ou menos épocas; com mais dados, a rapidez do clipping deixa
   de ser um problema.

## 8. Onde isso aparece no projeto

| Peça | Onde |
|------|------|
| `train_model` com `--warmup` | `sprints/ch05/experiments/treino_modelo.py` e o `exp06` do Cap. 5 |
| `clip_grad_norm_` | `train_model` (depois do warmup) |
| Laço de treino base | `train_model_simple` (Cap. 5), `treinar_rede` (Apêndice A) |
