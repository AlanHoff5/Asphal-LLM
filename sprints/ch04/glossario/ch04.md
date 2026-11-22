# Glossário — Capítulo 4: Implementando o Modelo GPT

Termos na ordem em que aparecem no capítulo.

---

**Transformer** — sem tradução: Arquitetura de rede neural baseada em atenção, criada em 2017. O GPT usa só o lado decoder (geração), sem encoder.

**Transformer Block** — Bloco Transformer: Unidade que se repete N vezes no GPT. Cada bloco aplica LayerNorm → Multi-Head Attention → atalho, depois LayerNorm → FeedForward → atalho. No GPT-2 small, 12 blocos empilhados; cada um tem 7.085.568 parâmetros.

**Layer Normalization (LayerNorm)** — Normalização de Camada: Ajusta cada vetor de token para média 0 e variância 1, depois aplica scale e shift aprendidos. Aparece duas vezes por bloco e uma vez ao final. Mantém os valores em escala estável durante o treino.

**Residual Connection / Shortcut Connection** — Conexão Residual / Atalho: Soma a entrada com a saída de cada sub-camada: `x + camada(x)`. Cria um caminho curto para o gradiente e preserva a informação original. Só funciona porque entrada e saída têm sempre 768 dimensões.

**Vanishing Gradient** — Desaparecimento do Gradiente: Em redes profundas, o gradiente encolhe a cada camada ao voltar. As conexões residuais e a LayerNorm são as principais defesas contra esse problema no GPT.

**FeedForward Network** — Rede Feed-Forward: Duas camadas lineares com GELU no meio. Expande o vetor de 768 → 3.072 e volta para 768. Opera em cada token independentemente e tem 4.722.432 parâmetros no GPT-2 small.

**GELU** (Gaussian Error Linear Unit): Função de ativação do GPT-2. Parecida com ReLU, mas suave para valores negativos (não zera, retorna um valor pequeno negativo). Mantém gradiente nos neurônios negativos, ajudando o treino.

**Dropout** — sem tradução: Zera aleatoriamente uma fração das unidades durante o treino. No GPT aparece três lugares: após os embeddings, nos pesos de atenção e antes de cada atalho. Desligado em `.eval()`.

**Logits** — sem tradução: Saída da última camada do modelo — um número por palavra do vocabulário por posição. Quanto maior, mais provável aquela palavra ser a próxima. Ainda não são probabilidades (não somam 1).

**Softmax**: Converte logits em probabilidades (positivos, somam 1). Usado na geração para sortear a próxima palavra e dentro da atenção (capítulo 3).

**Language Model Head (LM Head)** — Cabeça do Modelo de Linguagem: Camada linear final que transforma o vetor de 768 dimensões em 50.257 logits (um por token do vocabulário). Sem bias. No GPT-2 124M tem 38.597.376 parâmetros (~24% do modelo).

**Weight Tying** — Compartilhamento de Pesos: Usar a mesma matriz para o embedding de tokens e para o LM Head. Reduz parâmetros de 163M para 124M. O livro não usa no capítulo 4, mas carrega com weight tying no capítulo 5.

**Autoregressive Generation** — Geração Autoregressiva: Gera um token de cada vez: prevê o próximo, anexa ao texto, repete. Usa só os logits da última posição (a única que viu toda a sequência, por causa da máscara causal).

**Greedy Decoding** — Decodificação Gulosa: Escolhe sempre o token com maior logit (`argmax`). Rápido e determinístico, mas tende a gerar textos repetitivos.

**Temperature** — Temperatura: Divide os logits antes do softmax. Valor < 1 concentra a distribuição (mais previsível); valor > 1 achata (mais variado). Temperatura → 0 equivale à decodificação gulosa.

**Top-k Sampling** — Amostragem Top-k: Mantém só os k tokens com maiores logits; descarta o resto (logit = −∞). Usado junto com temperatura para equilibrar variedade e coerência.

**Inference** — Inferência: Usar o modelo para prever sem atualizar pesos. Requer `model.eval()` (desliga dropout) e `torch.no_grad()` (sem gradientes). É o contrário do treinamento (capítulo 5).

---

*17 termos — Capítulo 4 do livro Build a Large Language Model (From Scratch).*