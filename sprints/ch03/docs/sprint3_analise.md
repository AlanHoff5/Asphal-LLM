# Sprint 3 - Mecanismos de Atenção

## 1. Objetivo

A Sprint 3 substitui a limitação central da Sprint 2: os `Input Embeddings`
produzidos até ali representam cada token isoladamente (mais sua posição),
sem nenhuma relação entre os tokens da sequência. O mecanismo de atenção
resolve isso, permitindo que cada posição "consulte" as demais e construa
uma representação contextual - o `Context Vector`. O pipeline implementado:

```text
Input Embeddings (Sprint 2)
  ↓
Self-Attention sem pesos (3.3)
  ↓
Self-Attention com pesos treináveis / Scaled Dot-Product (3.4)
  ↓
Causal Attention (mascarada, em lote) (3.5)
  ↓
Multi-Head Attention (3.6)
  ↓
Entrada do Transformer Block (Sprint 4)
```

## 2. Componentes implementados

### Self-Attention sem pesos treináveis (`src/attention/self_attention.py`)

`attention_scores_simples()` reproduz a versão introdutória do capítulo:
o `Attention Score` entre dois tokens é o produto escalar de seus próprios
vetores de entrada, e o `Context Vector` de cada token é a soma dos vetores
de toda a sequência, ponderada pelos `Attention Weights` (scores
normalizados por softmax). Validamos essa implementação contra o valor de
referência publicado no livro para o vetor de contexto do token "journey"
no exemplo `"Your journey starts with one step"` - bateu exatamente:
`[0.4419, 0.6515, 0.5683]` (ver `tests/test_self_attention.py`).

### Self-Attention com pesos treináveis (`SelfAttention_v1` / `SelfAttention_v2`)

Introduz `Query`, `Key` e `Value`: cada token é projetado por três matrizes
de pesos treináveis, em vez de usar o embedding bruto como no passo
anterior. `SelfAttention_v1` usa `nn.Parameter` (fiel ao livro,
pedagógica); `SelfAttention_v2` usa `nn.Linear` (inicialização mais
estável, é a versão reaproveitada dali em diante). Ambas aplicam
`Scaled Dot-Product Attention`: os scores são divididos por
`sqrt(d_out)` antes do softmax.

### Causal Attention (`CausalAttention`)

Estende a self-attention para lotes (`batch_size, num_tokens, d_in`) e
adiciona a máscara causal: usando `torch.triu` com `diagonal=1`, marcamos
todas as posições futuras e as substituímos por `-inf` antes do softmax,
de modo que `softmax(-inf) = 0`. Também adiciona `Dropout` sobre os pesos
de atenção, ativo apenas em modo de treinamento (`.train()`).

### Multi-Head Attention (`MultiHeadAttentionWrapper` / `MultiHeadAttention`)

`MultiHeadAttentionWrapper` empilha várias `CausalAttention` independentes
e concatena as saídas (versão didática, um laço Python por cabeça).
`MultiHeadAttention` calcula o equivalente em uma única passagem
tensorial: projeta `Query/Key/Value` uma vez com `d_out` completo, divide
essa dimensão em `num_heads` fatias de `head_dim = d_out / num_heads` via
`view` + `transpose`, e processa todas as cabeças em paralelo. Ao final,
concatena as cabeças e aplica uma projeção linear extra (`out_proj`). Esta
é a versão que será reaproveitada no Transformer Block da Sprint 4.

## 3. Respostas técnicas

### Por que dividir os scores por `sqrt(d_out)`?

Sem o escalonamento, a magnitude do produto escalar cresce com a dimensão
`d_out` (mais termos sendo somados). Em dimensões altas, scores muito
grandes levam o softmax a se comportar quase como uma função degrau (um
peso perto de 1, os demais perto de 0), o que produz gradientes próximos
de zero e trava o treinamento. Reproduzimos esse efeito no Experimento 1
(seção 4): simulando scores de magnitude típica de `d_out=100`, o maior
peso médio por linha caiu de **0,789** (sem escala) para **0,401** (com
escala) - a distribuição fica visivelmente menos concentrada.

### Qual é o papel de Query, Key e Value?

`Query` representa o que o token atual está "procurando"; `Key` é o que
cada token "oferece" para ser encontrado por uma Query; `Value` é o
conteúdo que efetivamente entra no `Context Vector`, uma vez que sua
relevância (via `Key`) foi avaliada. Query e Key juntas decidem *quanto*
atenção prestar; Value decide *o que* é de fato somado ao resultado.

### Por que a atenção precisa ser causal?

O GPT é treinado por `Next-Word Prediction`: prever o próximo token dado
todo o contexto anterior. Se um token pudesse "ver" tokens futuros durante
o treinamento, o modelo estaria trapaceando - aprenderia a copiar a
resposta em vez de prevê-la. No Experimento 6, medimos isso diretamente:
sem máscara, a soma dos pesos que o primeiro token da sequência atribui
aos tokens seguintes (posições "futuras" para ele) é **0,83**; com a
máscara causal, essa soma cai para **0,0** - nenhuma informação futura
vaza para trás.

### O que muda entre Self-Attention (1 cabeça) e Multi-Head Attention?

Nada muda na contagem de parâmetros nem na dimensão final de saída, se o
`d_out` total for o mesmo (Experimento 5: ambas com 264 parâmetros e saída
`(1, 6, 8)`). O que muda é *como* o espaço de representação é dividido
internamente: com 1 cabeça, o modelo aprende um único "padrão" de relação
entre tokens; com múltiplas cabeças, cada uma opera em um subespaço menor
(`head_dim = d_out / num_heads`) e pode se especializar em um aspecto
diferente da sequência - relações sintáticas em uma cabeça, semânticas em
outra, por exemplo. É por isso que Multi-Head Attention captura relações
mais ricas do que uma única cabeça do mesmo tamanho total.

### Qual é a relação entre `d_out`, `num_heads` e `head_dim`?

`head_dim = d_out / num_heads`: por isso `d_out` precisa ser divisível por
`num_heads` (nossa implementação levanta `ValueError` caso contrário - ver
`test_d_out_precisa_ser_divisivel_por_num_heads`). No Experimento 3,
fixamos `d_out=24` e variamos `num_heads` de 1 a 6: a contagem de
parâmetros e a forma da saída não mudam (sempre 1176 parâmetros, saída
`(1, 6, 24)`) - só o `head_dim` interno muda (24, 12, 8, 6, 4). Já no
Experimento 4, fixamos `num_heads=4` e aumentamos `head_dim`: aí sim o
`d_out` total e a contagem de parâmetros crescem proporcionalmente (de 264
até 5696 parâmetros).

### O que a Sprint 4 vai usar daqui?

A saída da `MultiHeadAttention`, com o mesmo shape de entrada
`(batch_size, num_tokens, embedding_dim)`, é o `Context Vector` completo
da sequência - agora carregando informação relacional entre os tokens,
não só conteúdo e posição isolados. A Sprint 4 vai envolver essa saída em
um `Transformer Block` (com `Layer Normalization`, `Feed Forward Network`
e `Residual Connections`), empilhado várias vezes, para formar o modelo
GPT completo.

## 4. Experimentos

Todos os experimentos abaixo estão implementados e reproduzíveis em
[`experiments/fluxo_sprint3.py`](../experiments/fluxo_sprint3.py):

```powershell
python sprints/ch03/experiments/fluxo_sprint3.py
```

1. **Atenção com e sem escala** - compara o softmax dos mesmos scores
   brutos, com e sem dividir por `sqrt(d_out)`. Resultado: maior peso médio
   por linha cai de 0,789 para 0,401 com o escalonamento.
2. **Diferentes dimensões de embedding** (`d_out` de 2 a 128, self-attention
   de 1 cabeça) - a contagem de parâmetros cresce proporcionalmente
   (18 a 1152 parâmetros), a forma da saída acompanha `d_out`.
3. **Diferentes números de heads** (`d_out=24` fixo, `num_heads` de 1 a 6) -
   parâmetros e shape da saída não mudam; só o `head_dim` interno.
4. **Diferentes dimensões de head** (`num_heads=4` fixo, `head_dim` de 2 a
   16) - `d_out` total e parâmetros crescem junto com `head_dim`
   (264 a 5696 parâmetros).
5. **Self-Attention (1 cabeça) vs Multi-Head Attention** (mesmo `d_out=8`) -
   mesma contagem de parâmetros (264) e mesma forma de saída; a diferença
   é conceitual (um padrão de relação vs vários em paralelo).
6. **Comportamento da máscara causal** - soma dos pesos "futuros" cai de
   0,83 (sem máscara) para 0,0 (com máscara).
7. **Diferentes sequências de entrada** - três frases do corpus de
   Segurança da Informação do projeto, tokenizadas e processadas pelo
   pipeline completo (tokenizador → embeddings → multi-head attention).
8. **Visualização das matrizes de atenção** - heatmap por cabeça (4
   cabeças, sequência "Um firewall bloqueia ataques que tentam"), salvo em
   [`experiments/resultados/matriz_de_atencao.png`](../experiments/resultados/matriz_de_atencao.png).
   O formato triangular inferior é visível em todas as cabeças (consequência
   direta da máscara causal), e a primeira posição ("Um") tem peso 1,0 nela
   mesma, já que é o único token que ela pode "ver".

## 5. Conclusão

A Sprint 3 entrega o mecanismo que transforma embeddings isolados em
representações contextuais: cada token agora carrega informação sobre sua
relação com os tokens anteriores da sequência, de forma compatível com a
tarefa de `Next-Word Prediction` (graças à máscara causal) e capaz de
capturar múltiplos aspectos dessa relação em paralelo (graças às múltiplas
cabeças). O resultado - `Context Vectors` com shape
`(batch_size, num_tokens, embedding_dim)` - é o que a Sprint 4 vai envolver
em um `Transformer Block` completo para montar a arquitetura GPT.
