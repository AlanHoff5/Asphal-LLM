# Sprint 2 - Respostas Técnicas

Este documento relaciona os conceitos estudados no Capítulo 2 com a implementação existente no projeto. O fluxo analisado é:

```text
Texto
  -> tokenize()
  -> vocabulário e Token IDs
  -> GPTDatasetV1
  -> DataLoader
  -> GPTInputEmbeddings
  -> entrada do Transformer
```

## 1. Por que um LLM não pode trabalhar diretamente com o texto bruto?

Uma rede neural trabalha com operações numéricas sobre tensores. O texto bruto é formado por caracteres e palavras, possui comprimento variável e não pode ser usado diretamente como índice de uma camada neural.

No projeto, a função `tokenize()` transforma o texto em unidades menores. Depois, o vocabulário converte cada token em um inteiro. Por exemplo, a frase:

```text
Firewall bloqueia ataques.
```

é dividida em tokens semelhantes a:

```text
["Firewall", "bloqueia", "ataques", "."]
```

A sequência textual passa, então, a ser uma sequência de Token IDs que pode ser armazenada em tensores PyTorch.

## 2. Qual é a função do vocabulário?

O vocabulário é a tabela que relaciona cada token a um identificador numérico. Na implementação, `build_vocab()` coleta os tokens únicos, ordena-os e cria um dicionário no formato:

```python
{
    "token": id,
}
```

O tokenizador também cria a tabela inversa, permitindo recuperar o token a partir do ID. Essa relação é essencial para as duas direções do processamento:

```text
texto -> tokens -> Token IDs -> modelo
modelo -> Token IDs -> tokens -> texto
```

O vocabulário deve permanecer consistente durante o treinamento e a geração. Se um mesmo ID passasse a representar tokens diferentes, o modelo perderia a interpretação dos dados aprendida anteriormente.

## 3. Qual é a diferença entre um token e um Token ID?

Um **token** é uma unidade textual. Pode ser uma palavra, parte de uma palavra ou pontuação, dependendo da estratégia de tokenização.

Um **Token ID** é o número associado a esse token no vocabulário. O token possui significado textual; o ID é somente um índice usado para localizar dados numericamente.

No projeto, `SimpleTokenizerV2.encode()` realiza a conversão de tokens para IDs, enquanto `decode()` faz o caminho inverso. Tokens desconhecidos são substituídos por `<|unk|>`, e `<|endoftext|>` representa uma fronteira especial entre textos.

## 4. Por que os Token IDs não são utilizados diretamente como representação semântica?

Os IDs são números atribuídos por organização do vocabulário, e não por significado. Portanto, o ID 10 não é necessariamente semanticamente mais próximo do ID 11 do que do ID 500.

Se esses inteiros fossem usados diretamente, a rede poderia interpretar de maneira inadequada as diferenças numéricas entre tokens. O ID serve para localizar uma linha em uma tabela, não para representar o conteúdo do token.

Por isso, a implementação utiliza `torch.nn.Embedding`: o ID consulta um vetor denso treinável, que pode aprender relações úteis entre tokens.

## 5. Qual é a função dos embeddings?

Embeddings transformam IDs discretos em vetores de números reais com dimensão fixa. Essa representação pode ser processada pelas camadas do Transformer e ajustada durante o treinamento.

Na classe `GPTInputEmbeddings`, a tabela `token_embedding` recebe Token IDs e produz vetores de dimensão `embedding_dim`. Para uma entrada com shape:

```text
(batch_size, context_length) = (2, 3)
```

com `embedding_dim=5`, o resultado dos vetores de token possui shape:

```text
(2, 3, 5)
```

Assim, cada posição da sequência deixa de ser apenas um inteiro e passa a ser representada por cinco valores aprendíveis.

## 6. Por que é necessário representar a posição dos tokens?

A identidade de um token não informa onde ele aparece na sequência. As frases abaixo possuem os mesmos tokens, mas ordens diferentes:

```text
"gato persegue rato"
"rato persegue gato"
```

A posição altera o significado. Por isso, `GPTInputEmbeddings` possui uma segunda tabela chamada `position_embedding`. Ela gera um vetor para cada posição da sequência, começando em zero.

A entrada final do modelo é calculada por:

```python
input_embedding = token_embedding + position_embedding
```

O token embedding representa o conteúdo; o positional embedding representa a posição. A soma mantém o shape `(batch_size, context_length, embedding_dim)` e combina as duas informações antes da atenção.

## 7. Qual é a relação entre tamanho do contexto e quantidade de amostras?

O `max_length` define quantos tokens existem em cada entrada. Para formar também o alvo deslocado, a sequência precisa possuir um token adicional. Se:

- `N` é a quantidade total de Token IDs;
- `L` é `max_length`;
- `S` é `stride`;

então, para o `range()` usado em `GPTDatasetV1`, a quantidade de amostras é:

$$
\text{amostras} =
\left\lfloor\frac{N-L-1}{S}\right\rfloor + 1
$$

quando `N > L`.

No teste do projeto:

```text
N = 8, L = 3, S = 1
```

São produzidas cinco janelas. Cada janela possui três tokens de entrada e três tokens de alvo, sendo o alvo deslocado uma posição:

```text
entrada: [um, dois, tres]
alvo:    [dois, tres, quatro]
```

Aumentar o contexto geralmente reduz a quantidade de janelas possíveis. Reduzir o stride aumenta a sobreposição e pode gerar mais amostras, mas também repete mais conteúdo entre janelas.

## 8. Qual é o impacto da dimensão do embedding sobre as estruturas utilizadas pelo modelo?

A dimensão do embedding é a quantidade de valores usada para representar cada token e cada posição.

Com uma entrada `(2, 3)`:

```text
embedding_dim=5  -> saída (2, 3, 5)
embedding_dim=64 -> saída (2, 3, 64)
```

A tabela de embeddings de tokens possui:

```text
vocab_size * embedding_dim
```

parâmetros. Portanto, aumentar a dimensão pode permitir representações mais expressivas, mas aumenta o consumo de memória, o custo de transferência dos dados e o processamento das camadas seguintes, incluindo a atenção.

A dimensão também precisa ser compatível com a arquitetura do Transformer que será construída nas próximas Sprints.

## 9. Qual é a função do DataLoader no pipeline?

O `GPTDatasetV1` cria e armazena os pares individuais de entrada e alvo. O `DataLoader` usa esse Dataset para agrupá-los em lotes, que são mais adequados para o processamento paralelo pela rede neural.

No projeto, `create_dataloader_v1()` recebe o texto e o tokenizador, cria o Dataset e devolve um `torch.utils.data.DataLoader`. Ele controla:

- `batch_size`: quantidade de amostras por lote;
- `shuffle`: se as amostras serão embaralhadas;
- `drop_last`: se o último lote incompleto será descartado;
- `num_workers`: quantidade de processos usados para carregar os dados.

No teste, cinco amostras com `batch_size=2` produzem três lotes quando `drop_last=False`. O último possui apenas uma amostra. Com `drop_last=True`, são mantidos dois lotes completos com duas amostras cada.

O DataLoader não tokeniza, não cria o vocabulário e não calcula embeddings. Sua responsabilidade é organizar as amostras já preparadas para o loop de treinamento.

## 10. Quais informações serão utilizadas pelo mecanismo de atenção?

A próxima Sprint receberá os embeddings de entrada produzidos pela soma entre os vetores de token e de posição. Para um lote com oito sequências, contexto quatro e dimensão de embedding 256, o formato será:

```text
input_embeddings: (8, 4, 256)
```

Esse tensor será usado pelo mecanismo de atenção para calcular relações entre as posições da mesma sequência.

Os `targets` não entram diretamente no cálculo da atenção. Eles permanecem como Token IDs e serão usados para comparar a previsão do modelo com o próximo token esperado durante o cálculo da função de perda.

Portanto, a Sprint 2 entrega para a próxima etapa:

```text
inputs  -> Token IDs organizados em lotes e depois convertidos em embeddings
targets -> Token IDs deslocados usados como referência de treinamento
```

## 11. Evidências experimentais do projeto

A execução de `tests/test_dataloader.py` confirma o fluxo com um texto de oito tokens:

```text
Dataset: 5 amostras
Lotes produzidos: 3
Ultimo lote tem 1 amostra(s)
Alvos deslocados corretamente: True

Lotes completos preservados: 2
Shapes: [(2, 3), (2, 3)]
```

Também foi verificada a integração com embeddings:

```text
inputs: (2, 3)
targets: (2, 3)
embeddings: (2, 3, 5)
```

Esses resultados demonstram que o texto é transformado em Token IDs, organizado em pares de treinamento, agrupado em lotes e convertido em vetores com informação de conteúdo e posição.
