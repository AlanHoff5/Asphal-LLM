# Glossário — Apêndice E: Fine-tuning eficiente em parâmetros com LoRA

Termos na ordem em que aparecem no apêndice.

---

**Parameter-Efficient Fine-Tuning (PEFT)** — Fine-tuning eficiente em parâmetros: Família de técnicas que adaptam um modelo pré-treinado treinando só uma fração pequena dos parâmetros. Gasta menos memória e gera arquivos de ajuste pequenos.

**LoRA (Low-Rank Adaptation)** — Adaptação de posto baixo: Congela os pesos W de cada camada linear e aprende uma correção ΔW = A · B formada por duas matrizes estreitas. No GPT-2 124M com rank 16, treina 2.666.528 parâmetros.

**Weight Update (ΔW)** — Atualização dos pesos: O que o fine-tuning soma aos pesos pré-treinados. No fine-tuning comum ΔW tem o tamanho de W; no LoRA ela é aproximada por A · B.

**Low-Rank Matrix** — Matriz de posto baixo: Matriz que pode ser escrita como o produto de duas matrizes estreitas. Uma matriz 768 × 768 de posto 16 cabe em 16 · (768 + 768) = 24.576 números.

**Rank** — Posto: Largura interna das matrizes A (entrada × rank) e B (rank × saída). Rank maior dá mais capacidade de ajuste e mais parâmetros; o livro usa 16.

**Alpha** — Alfa: Fator que escala a saída do LoRA. Aqui a escala é `alpha / rank`, como no repositório atual do livro; com alpha = rank, a escala é 1.

**Matrix A** — Matriz A: Primeira matriz do LoRA, de entrada × rank, iniciada como os pesos de uma `nn.Linear` comum (Kaiming uniforme).

**Matrix B** — Matriz B: Segunda matriz do LoRA, de rank × saída, iniciada com **zeros**. Por isso, antes do treino, o modelo com LoRA dá exatamente as mesmas saídas que o pré-treinado.

**LoRALayer** — Camada LoRA: Módulo que calcula `(alpha / rank) · x · A · B`, o desvio de posto baixo.

**LinearWithLoRA** — Linear com LoRA: Junta uma `nn.Linear` congelada e uma `LoRALayer`: `saída = linear(x) + lora(x)`.

**replace_linear_with_lora** — Substituir lineares por LoRA: Percorre o modelo recursivamente e troca cada `nn.Linear` por `LinearWithLoRA`. No projeto aceita uma lista de camadas a ignorar, como o `out_head` que a classificação não usa.

**Frozen Base Model** — Modelo base congelado: Todos os pesos do GPT-2 com `requires_grad = False`. No LoRA só A e B recebem gradientes; no cap. 6 o último bloco também era treinado.

**Adapter Weights** — Pesos do adaptador: Só os tensores A e B, salvos à parte. Bastam para recriar o modelo ajustado a partir do checkpoint pré-treinado e ocupam cerca de 1,6% do tamanho do modelo.

**Weight Merging** — Mesclagem de pesos: Soma `(alpha / rank) · (A · B)ᵀ` em W uma vez depois do treino. O modelo volta a ter só camadas `nn.Linear` e a inferência fica sem custo extra, com as mesmas saídas.

**Trainable Parameter Fraction** — Fração de parâmetros treináveis: Parâmetros treinados divididos pelos parâmetros do modelo original. Neste projeto: 4,35% no cap. 6 e 1,64% com LoRA rank 16.
