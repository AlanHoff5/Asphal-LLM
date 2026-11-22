# Apêndice A - Introdução ao PyTorch

## 1. Objetivo

O Apêndice A apresenta o PyTorch do zero: tensores, diferenciação
automática, redes com `torch.nn`, `Dataset`/`DataLoader`, o laço de treino,
salvar e carregar pesos e o uso da GPU. Tudo o que os capítulos 2 a 7 usam
aparece aqui primeiro, num problema pequeno o bastante para ser conferido à
mão: 5 pontos de treino e 2 de teste, com 2 classes.

```text
dados_toy() → ToyDataset → DataLoader (lotes de 2, shuffle, drop_last)
  ↓
Laço de treino, para cada lote:
  NeuralNetwork(2, 2) → logits → cross-entropy → zero_grad → backward → SGD.step
  ↓
compute_accuracy (treino e teste) → torch.save(state_dict) → load_state_dict
```

O código reutilizável fica em `src/pytorch_basico/`:

| Arquivo | Conteúdo |
|---------|----------|
| `rede_neural.py` | `NeuralNetwork` (2 camadas ocultas: 30 e 20), contagem de parâmetros |
| `dados_toy.py` | `dados_toy`, `ToyDataset`, `criar_loaders_toy` |
| `treino_basico.py` | `treinar_rede` (laço da seção A.7) e `compute_accuracy` |

A escolha do dispositivo reaproveita `src/training/dispositivo.py` (Sprint 5).

**Ambiente dos resultados abaixo:** CPU de 2 threads, PyTorch 2.x, semente
123. Os tempos mudam de máquina para máquina; os valores de perda,
gradiente e acurácia não.

## 2. Tensores (seção A.2)

| Tensor | Shape | dtype |
|--------|-------|-------|
| `torch.tensor(1)` | `()` | `int64` |
| `torch.tensor([1, 2, 3])` | `(3,)` | `int64` |
| `torch.tensor([[1, 2], [3, 4]])` | `(2, 2)` | `int64` |
| `torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])` | `(2, 2, 2)` | `int64` |

Dois detalhes que voltam nos capítulos seguintes:

- `torch.tensor(array)` **copia** o array do NumPy, e `torch.from_numpy`
  **compartilha** a memória. Mudar o array depois muda só o segundo.
- A transposta `.T` não é contígua na memória: `t.T.view(6)` dá erro, e
  `t.T.reshape(6)` funciona. É por isso que a `MultiHeadAttention` do cap. 3
  chama `.contiguous()` antes do `view` que junta as heads.

## 3. Grafo computacional e autograd (seções A.3 e A.4)

Regressão logística de um neurônio com `x1 = 1,1`, `w1 = 2,2`, `b = 0` e
rótulo `y = 1`: `z = 2,42`, `a = sigmoid(z) = 0,9183` e perda BCE = 0,0852.

| Gradiente | `autograd.grad` | `.backward()` | À mão: `(a − y)·x` |
|-----------|-----------------|---------------|--------------------|
| dL/dw1 | −0,0898 | −0,0898 | −0,0898 |
| dL/db | −0,0817 | −0,0817 | −0,0817 |

Os três caminhos dão o mesmo valor, e o valor é o do livro. Para sigmoid
com entropia cruzada binária a derivada em relação a `z` é só `a − y`, o
que torna a conta à mão curta. Os gradientes são negativos porque aumentar
`w1` aproxima `a` de 1 e diminui a perda.

Um teste extra mostra por que todo laço de treino chama `zero_grad()`:
duas chamadas de `backward()` seguidas **somam** os gradientes (3 + 3 = 6).

## 4. Rede multicamadas (seção A.5)

`NeuralNetwork(50, 3)` tem **2.213 parâmetros treináveis**, conferidos à mão:
`(50·30 + 30) + (30·20 + 20) + (20·3 + 3)`. Com a semente 123 e uma
entrada aleatória, os logits são `[−0,1262; 0,1080; −0,1792]`, iguais ao
livro, e a softmax os transforma em `[0,311; 0,393; 0,295]`.

A rede devolve logits, não probabilidades. A softmax fica fora do modelo
porque `F.cross_entropy` aplica log-softmax internamente; aplicá-la duas
vezes deixaria a perda instável. O `GPTModel` segue a mesma regra.

## 5. Dataset, DataLoader e treino (seções A.6 e A.7)

Com 5 exemplos e lotes de 2, o `DataLoader` gera lotes de 2, 2 e 1. Com
`drop_last=True` o lote de 1 exemplo é descartado e sobram 2 lotes por
época.

O treino do livro (SGD, lr = 0,5, 3 épocas) reproduz exatamente a sequência
de perdas publicada:

| Época | Lote 1 | Lote 2 |
|-------|--------|--------|
| 1 | 0,75 | 0,65 |
| 2 | 0,44 | 0,13 |
| 3 | 0,03 | 0,00 |

A acurácia final é **100% no treino e 100% no teste**.

## 6. Salvar, carregar e GPU (seções A.8 e A.9)

O `state_dict` tem 6 tensores (`layers.0`, `layers.2` e `layers.4`, cada um
com peso e viés) e ocupa 5.885 bytes. Depois de `load_state_dict` numa
`NeuralNetwork(2, 2)` nova, as saídas são idênticas; numa arquitetura
diferente, o carregamento falha com `RuntimeError`.

Treinar na GPU exige só três mudanças no laço: escolher o `device`, chamar
`model.to(device)` e mover cada lote com `.to(device)`. O treino no
dispositivo disponível chega aos mesmos 100% de acurácia.

O treino com várias GPUs (DDP, seção A.9.3) não foi reproduzido: ele
exige duas ou mais GPUs e o backend NCCL, que não existe no Windows.

## 7. Experimentos

Script: `experiments/fluxo_apendiceA.py` → `resultados/apendiceA_resultados.json`.

### 7.1 Taxa de aprendizado (SGD, 10 épocas)

| lr | Perda no 1º passo | Perda final | Acurácia de teste |
|----|-------------------|-------------|-------------------|
| 0,005 | 0,7487 | 0,5977 | 100% |
| 0,05 | 0,7487 | 0,0787 | 100% |
| 0,5 | 0,7487 | 0,0016 | 100% |
| 5,0 | 0,7487 | 6,2413 | 50% |

Com lr = 0,005 a perda quase não sai do lugar em 10 épocas. Com lr = 5,0
os passos passam do mínimo e a perda **diverge**: a rede passa a prever
uma classe só e acerta 50%. É o mesmo efeito que o cap. 5 mostra no GPT
(exp02, lr alta) e que o Apêndice D ataca com warmup.

### 7.2 Tamanho do lote (SGD, lr = 0,5, 3 épocas)

| batch_size | Lotes por época | Passos | Perda final | Acurácia de teste |
|------------|-----------------|--------|-------------|-------------------|
| 1 | 5 | 15 | 0,0111 | 100% |
| 2 | 2 | 6 | 0,0043 | 100% |
| 5 | 1 | 3 | 0,1345 | 100% |

Lote 5 (o conjunto inteiro) faz só 3 atualizações e termina com a maior
perda. Lote 1 faz 15 atualizações, mas cada uma vê um exemplo só e é mais
ruidosa.

### 7.3 CPU × GPU

Uma multiplicação 2048 × 2048 levou 61,3 ms na CPU de 2 threads do
ambiente de verificação. Rode o script na máquina com GPU para preencher a
comparação: o resultado aparece na mesma seção do JSON.

## 8. Limitações

- O problema é minúsculo de propósito: 7 pontos separáveis por uma reta.
  Ele serve para conferir cada peça, não para medir generalização.
- O experimento de CPU × GPU mede uma operação isolada. Num treino real,
  copiar dados para a GPU também custa tempo.
- O DDP fica só descrito, pelos motivos da seção 6.

## 9. Onde isso aparece no projeto

| Conceito do Apêndice A | Onde é usado |
|------------------------|--------------|
| `Dataset` + `DataLoader` | `GPTDatasetV1` e `create_dataloader_v1` (cap. 2), `SpamDataset` (cap. 6) |
| `nn.Module` + `nn.Sequential` | `MultiHeadAttention`, `FeedForward`, `GPTModel` (caps. 3 e 4) |
| Laço de treino | `train_model_simple` (cap. 5), `treinar_classificador` (cap. 6) |
| `requires_grad = False` | congelamento de camadas (cap. 6) e LoRA (Apêndice E) |
| `state_dict` | checkpoints (cap. 5) e pesos do GPT-2 (cap. 6) |
| `device` | `escolher_dispositivo` (cap. 5 em diante) |
