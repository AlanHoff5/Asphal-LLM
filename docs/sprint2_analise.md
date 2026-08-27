# Sprint 2 - Trabalhando com Dados Textuais

## 1. Objetivo

A Sprint 2 transforma texto bruto em tensores que podem ser recebidos por um modelo de linguagem. O pipeline implementado é:

```text
Texto
  ↓
Tokenização
  ↓
Vocabulário e Token IDs
  ↓
Sequências de treinamento
  ↓
Token Embeddings
  ↓
Positional Embeddings
  ↓
Lote de dados
  ↓
Entrada do Transformer
```

## 2. Componentes implementados

### Tokenização

`src/data/tokenizer.py` divide o texto usando uma expressão regular que preserva palavras, pontuação e o marcador `--`. A função `load_text()` lê o corpus UTF-8 da CompTIA Security+.

### Vocabulário e Token IDs

`build_vocab()` coleta os tokens únicos, ordena-os e associa cada token a um inteiro. O dicionário permite a relação bidirecional entre token e ID. Os tokens `<|unk|>` e `<|endoftext|>` são adicionados ao vocabulário.

`SimpleTokenizerV1` falha para tokens ausentes. `SimpleTokenizerV2` substitui tokens ausentes por `<|unk|>`. Para BPE, `get_bpe_tokenizer()` usa o vocabulário GPT-2 da biblioteca `tiktoken`.

### Sequências de treinamento

`GPTDatasetV1` transforma a sequência de IDs em janelas de tamanho `max_length`. O alvo é a entrada deslocada em uma posição, pois o modelo aprende a prever o próximo token.

Exemplo:

```text
entrada: [40, 367, 2885, 1464]
alvo:    [367, 2885, 1464, 1807]
```

O parâmetro `stride` controla quanto a janela avança e, portanto, quanto as amostras se sobrepõem.

### Embeddings

`GPTInputEmbeddings` usa duas tabelas `torch.nn.Embedding`: uma para o conteúdo dos tokens e outra para suas posições. A soma das duas produz o vetor de entrada:

```text
input_embedding = token_embedding + positional_embedding
```

Para um lote com shape `(batch_size, context_length)` e dimensão `embedding_dim`, a saída possui shape `(batch_size, context_length, embedding_dim)`.

### DataLoader

`create_dataloader_v1()` usa o `GPTDatasetV1` e o `torch.utils.data.DataLoader` para agrupar pares de entrada e alvo em lotes. O Transformer recebe esses lotes depois que os Token IDs passam pela camada de embeddings.

## 3. Respostas técnicas

### Por que o LLM não trabalha diretamente com texto bruto?

As operações da rede neural trabalham com tensores numéricos. Texto bruto possui símbolos discretos e comprimento variável; a tokenização cria unidades manipuláveis e os Token IDs fornecem índices inteiros para as tabelas de embedding.

### Qual é a função do vocabulário?

O vocabulário define o conjunto de tokens conhecidos e a correspondência entre cada token e seu ID. Ele precisa permanecer o mesmo durante treinamento e geração para que um ID continue representando o mesmo token.

### Qual é a diferença entre token e Token ID?

Token é a unidade textual, como uma palavra ou pontuação. Token ID é o número associado a essa unidade no vocabulário. O ID é um índice, não uma representação semântica.

### Por que Token IDs não são usados diretamente como representação semântica?

A distância entre inteiros não representa significado. Por exemplo, os IDs 10 e 11 não precisam ser semanticamente mais próximos que 10 e 500. A camada de embedding aprende vetores densos que podem representar relações úteis para a rede.

### Qual é a função dos embeddings?

Embeddings convertem IDs discretos em vetores de dimensão fixa. Esses vetores são treináveis e formam a representação contínua processada pelas camadas do Transformer.

### Por que representar a posição dos tokens?

A atenção opera sobre os elementos da sequência, mas não fornece sozinha uma noção absoluta de ordem. O positional embedding informa em qual posição cada token aparece. A soma com o token embedding produz a entrada completa.

### Qual é a relação entre contexto e quantidade de amostras?

Para uma sequência com `N` tokens e contexto `L`, avançando `S` posições, aproximadamente:

$$
\text{amostras} = \left\lfloor\frac{N-L}{S}\right\rfloor + 1
$$

Um contexto maior produz menos janelas. Um stride menor aumenta a sobreposição e produz mais amostras, mas repete mais conteúdo.

### Qual é o impacto da dimensão do embedding?

A saída passa a ter uma dimensão maior, por exemplo, de `(8, 4, 64)` para `(8, 4, 256)`. A tabela de token embedding possui `vocab_size × embedding_dim` parâmetros. Aumentar essa dimensão pode aumentar a capacidade de representação e também o custo de memória e processamento.

### Qual é a função do DataLoader?

O DataLoader agrupa amostras do Dataset em batches, controla o embaralhamento e pode descartar o último lote incompleto. Isso permite que o treinamento processe várias sequências de forma organizada.

### O que será usado pelo mecanismo de atenção?

A próxima Sprint usará os `input_embeddings` com shape `(batch_size, context_length, embedding_dim)`. Os alvos continuarão sendo Token IDs para calcular a perda da previsão do próximo token.

## 4. Experimentos

Os experimentos do notebook `notebooks/InteligenciaArtificial_SistemasInteligentes.ipynb` variam:

- quantidade de tokens para textos diferentes;
- tokenização por palavras e BPE;
- tamanho do contexto;
- stride e sobreposição;
- tamanho do batch;
- dimensão dos embeddings;
- custo de tokenização.

Os testes automatizados em `tests/test_sprint2.py` verificam as propriedades estruturais do pipeline. Os valores numéricos dos experimentos devem ser obtidos executando as células correspondentes do notebook no ambiente configurado.

## 5. Conclusão

A Sprint 2 entrega a preparação dos dados até a representação vetorial com posição. O resultado que seguirá para a Sprint 3 é um lote de embeddings de entrada; a saída esperada mantém a ordem dos tokens e informa ao mecanismo de atenção tanto o conteúdo quanto a posição de cada elemento.
