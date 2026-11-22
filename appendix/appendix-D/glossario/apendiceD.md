# Glossário — Apêndice D: Incrementando o laço de treino

Termos na ordem em que aparecem no apêndice.

---

**Learning Rate Schedule** — Cronograma da taxa de aprendizado: Regra que muda a taxa de aprendizado a cada passo do treino, em vez de mantê-la fixa. O Apêndice D combina warmup e cosine decay.

**Learning Rate Warmup** — Aquecimento da taxa de aprendizado: Começa o treino com uma taxa pequena (`initial_lr`) e sobe em linha reta até o pico (`peak_lr`). Evita passos grandes enquanto os pesos ainda são aleatórios e os gradientes pouco confiáveis.

**Initial Learning Rate** — Taxa de aprendizado inicial: Valor do primeiro passo do warmup. O livro usa 0,0001 na demonstração e 0,00001 no treino da seção D.4.

**Peak Learning Rate** — Taxa de aprendizado de pico: Maior taxa do treino, alcançada no fim do warmup. Em `train_model` ela é o `lr` configurado no otimizador.

**Warmup Steps** — Passos de aquecimento: Quantos passos o warmup dura. O livro usa 20% do total: com 9 lotes por época e 15 épocas, `int(0,2 × 135) = 27`.

**Global Step** — Passo global: Contador de atualizações dos pesos desde o início do treino, somando todas as épocas. É ele que define em que ponto do cronograma o treino está.

**Cosine Decay** — Decaimento em cosseno: Depois do pico, a taxa desce em meia onda de cosseno até `min_lr`: devagar no começo, mais rápido no meio e devagar de novo no fim. Também chamado de *cosine annealing*.

**Minimum Learning Rate** — Taxa de aprendizado mínima: Valor final do cosine decay. Passos finos no fim do treino ajustam os pesos sem desfazer o que já foi aprendido.

**Gradient Clipping** — Corte do gradiente: Limita o tamanho dos gradientes antes do passo do otimizador, para que um lote atípico não cause uma atualização enorme. O livro só aplica depois do warmup.

**Gradient Norm (L2)** — Norma L2 do gradiente: Raiz da soma dos quadrados de todos os gradientes do modelo, tratados como um vetor só. É a medida que o `clip_grad_norm_` compara com o limite.

**clip_grad_norm_** — Corte pela norma: Função do PyTorch que, se a norma L2 passar de `max_norm`, multiplica todos os gradientes por `max_norm / norma`. A direção do passo continua a mesma; só o tamanho diminui. Ela devolve a norma original.

**max_norm** — Norma máxima: Limite do gradient clipping. O livro usa 1,0.

**Exploding Gradients** — Gradientes explosivos: Gradientes que ficam muito grandes de repente e fazem a perda saltar. O clipping limita o gradiente, mas com o AdamW o tamanho do passo também depende da média móvel dos gradientes.

**Training Stability** — Estabilidade do treino: Quanto a perda cai de forma suave, sem saltos nem divergência. Warmup, cosine decay e clipping existem para melhorá-la, sobretudo com taxas de aprendizado altas.

**train_model** — Laço de treino modificado: Função de `src/training/treino.py` que soma ao `train_model_simple` do cap. 5 o cronograma de warmup e cosine e o gradient clipping, e registra a taxa usada em cada passo.
