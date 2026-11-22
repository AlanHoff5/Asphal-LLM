# Glossário — Apêndice A: Introdução ao PyTorch

Termos na ordem em que aparecem no apêndice.

---

**PyTorch** — PyTorch: Biblioteca de deep learning em Python com três partes: tensores (como o NumPy, mas com GPU), diferenciação automática e ferramentas para montar e treinar redes (`torch.nn`, otimizadores, `DataLoader`). Todo o projeto é escrito sobre ela.

**Tensor** — Tensor: Arranjo de números com qualquer número de dimensões. Escalar = 0D, vetor = 1D, matriz = 2D; um lote de embeddings do GPT é um tensor 3D `(batch, tokens, emb_dim)`.

**Data Type (dtype)** — Tipo de dado: Formato numérico dos elementos de um tensor. Inteiros viram `int64` e números com vírgula viram `float32`, que tem precisão suficiente e é mais rápido na GPU que `float64`.

**Reshape / View** — Remodelar / Visão: Mudam o formato de um tensor sem mudar os dados. `view` exige memória contígua e falha numa transposta; `reshape` copia quando precisa.

**Matrix Multiplication** — Multiplicação de matrizes: Operação `A @ B` (ou `A.matmul(B)`). É a conta mais usada num LLM: projeções Q, K, V, pontuações de atenção e camadas lineares.

**Computation Graph** — Grafo computacional: Registro das operações feitas sobre tensores, da entrada até a perda. O PyTorch o monta durante o forward e o percorre de trás para frente para calcular gradientes.

**Autograd** — Diferenciação automática: Motor do PyTorch que aplica a regra da cadeia sobre o grafo computacional. Dispensa derivar cada camada à mão.

**Gradient** — Gradiente: Derivada parcial da perda em relação a cada parâmetro. Indica para que lado e quanto mudar um peso para a perda diminuir.

**requires_grad** — Requer gradiente: Marca um tensor para que o PyTorch guarde o grafo e calcule seu gradiente. Parâmetros de `nn.Module` já vêm com `True`; congelar uma camada é colocar `False` (cap. 6).

**Backpropagation (backward)** — Retropropagação: Cálculo dos gradientes de trás para frente pelo grafo. `loss.backward()` preenche o `.grad` de cada parâmetro, e chamadas seguidas **somam** os gradientes.

**Multilayer Perceptron** — Perceptron multicamadas: Rede de camadas lineares intercaladas por funções de ativação. A do apêndice tem duas camadas ocultas (30 e 20 neurônios) e 2.213 parâmetros para 50 entradas e 3 saídas.

**nn.Module** — Módulo: Classe base de toda camada e modelo no PyTorch. Define as camadas no `__init__` e o cálculo no `forward`; o `GPTModel` do cap. 4 também é um `nn.Module`.

**nn.Sequential** — Sequencial: Contêiner que aplica camadas em ordem. Usado na `NeuralNetwork`, na `FeedForward` e nos `trf_blocks` do GPT.

**ReLU** — Unidade linear retificada: Ativação `max(0, x)`. O GPT usa a GELU, uma versão suave dela (cap. 4).

**Logits** — Logits: Saídas da última camada antes de qualquer normalização. Podem ser negativos e não somam 1.

**Softmax** — Softmax: Converte logits em probabilidades que somam 1. Fica fora do modelo porque `F.cross_entropy` já a aplica internamente.

**Cross-Entropy Loss** — Perda de entropia cruzada: Mede o quanto a probabilidade prevista para a classe certa está longe de 1. É a mesma perda do pré-treino (cap. 5) e do fine-tuning (caps. 6 e 7).

**Dataset** — Conjunto de dados: Classe com `__getitem__` (um exemplo) e `__len__` (quantos exemplos). `ToyDataset` aqui, `GPTDatasetV1` no cap. 2 e `SpamDataset` no cap. 6.

**DataLoader** — Carregador de dados: Monta lotes a partir de um `Dataset`, embaralha a cada época (`shuffle`) e pode usar processos paralelos (`num_workers`).

**drop_last** — Descartar o último: Opção do `DataLoader` que descarta o lote final incompleto. Um lote de 1 exemplo gera um gradiente muito ruidoso.

**Training Loop** — Laço de treino: Para cada lote: forward, perda, `zero_grad`, `backward` e `step`. É o mesmo esqueleto do `train_model_simple` do cap. 5.

**Optimizer (SGD)** — Otimizador (gradiente descendente estocástico): Atualiza os pesos com `w = w − lr · gradiente`. O livro usa SGD aqui e AdamW no GPT.

**Learning Rate** — Taxa de aprendizado: Tamanho do passo do otimizador. No experimento do apêndice, lr = 0,005 quase não treina e lr = 5,0 faz a perda divergir.

**Epoch** — Época: Uma passada completa por todos os lotes de treino.

**model.train() / model.eval()** — Modo de treino / avaliação: Liga ou desliga comportamentos que só existem no treino, como dropout. O GPT usa dropout, então a diferença importa a partir do cap. 4.

**torch.no_grad()** — Sem gradiente: Bloco em que o PyTorch não monta o grafo. Usado na avaliação e na geração de texto: mais rápido e com menos memória.

**state_dict** — Dicionário de estado: Dicionário `nome da camada → tensor` com todos os pesos do modelo. É o que `torch.save` grava e `load_state_dict` restaura.

**Device (CUDA)** — Dispositivo: Onde os tensores ficam e as contas acontecem: `cpu`, `cuda` (NVIDIA), `mps` (Apple) ou `xpu` (Intel). Modelo e lotes precisam estar no mesmo dispositivo.

**DistributedDataParallel (DDP)** — Paralelismo de dados distribuído: Treino em várias GPUs, cada uma com uma cópia do modelo e uma fatia dos lotes, com gradientes sincronizados. Exige o backend NCCL, que não existe no Windows.
