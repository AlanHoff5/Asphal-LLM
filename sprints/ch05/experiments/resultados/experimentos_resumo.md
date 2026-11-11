# Resumo dos experimentos de treinamento (Sprint 5)

Todos usam The Verdict (90% treino / 10% validacao), AdamW e seed 123. Cada experimento muda um parametro em relacao ao baseline.

| experimento | o que mudou | parametros | perda treino | perda val | menor val (epoca) | gap val-treino | ppl val | tempo | tokens/s | GPU (GB) | overfitting? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exp01_baseline | treino do livro | 162.4M | 0.420 | 6.452 | 6.087 (5.1) | 6.033 | 634.1 | 1.1 min | 686 | 3.07 | sim: val subiu 0.37 depois do minimo (gap 6.03) |
| exp02_lr_alta | lr 0.0004 -> 0.001 | 162.4M | 5.488 | 6.819 | 6.763 (2.3) | 1.331 | 915.1 | 1.1 min | 709 | 3.07 | gap grande (1.33) |
| exp03_lr_baixa | lr 0.0004 -> 0.0001 | 162.4M | 2.900 | 6.211 | 6.189 (9.6) | 3.312 | 498.3 | 1.4 min | 535 | 3.07 | gap grande (3.31) |
| exp04_30_epocas | 10 -> 30 epocas | 162.4M | 0.006 | 7.297 | 6.087 (5.1) | 7.291 | 1,475.1 | 4.4 min | 527 | 3.07 | sim: val subiu 1.21 depois do minimo (gap 7.29) |
| exp05_sem_regularizacao | dropout 0 e weight decay 0 | 162.4M | 0.661 | 6.352 | 6.132 (6.2) | 5.691 | 573.4 | 1.2 min | 667 | 3.07 | sim: val subiu 0.22 depois do minimo (gap 5.69) |
| exp06_warmup_cosine_clip | lr 0.001 + warmup 20% + cosine + clip 1.0 | 162.4M | 1.217 | 6.230 | 6.180 (6.2) | 5.013 | 507.8 | 1.1 min | 724 | 3.07 | gap grande (5.01) |
| exp07_modelo_mini | modelo mini (emb 128, 4 camadas, contexto 128) | 13.7M | 5.411 | 6.445 | 6.445 (10.0) | 1.034 | 629.7 | 0.3 min | 2,463 | 0.34 | gap grande (1.03) |
| exp08_modelo_small | modelo small (emb 256, 6 camadas, contexto 256) | 30.5M | 4.492 | 6.229 | 6.229 (10.0) | 1.737 | 507.3 | 0.3 min | 2,463 | 0.80 | gap grande (1.74) |

## Ultima amostra de texto de cada experimento

Contexto inicial: "Every effort moves you" (decodificacao gulosa).

- **exp01_baseline** (epoca 10): Every effort moves you?"     I glanced after him, so inevitably the background of the house."      He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I
- **exp02_lr_alta** (epoca 10): Every effort moves you, the, the, the, the, the, the I,, the, the, the, the, the I, the I, the, the, the, the, the, the, the, the, the, the, the
- **exp03_lr_baixa** (epoca 10): Every effort moves you know. "I was a he was to the of the his of a.        "I to the.  I that he had been his. I, and, and--I. "I
- **exp04_30_epocas** (epoca 30): Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I
- **exp05_sem_regularizacao** (epoca 10): Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"   "I moved away, instinctively embarrassed by my unexpected discovery; and as I turned, when Stroud.   
- **exp06_warmup_cosine_clip** (epoca 10): Every effort moves you?"  "Yes--quite insensible to the irony. Gisburn's an--and by me!"  "Oh, and I felt back his head to look up at the honour of the donkey. "There were days when I
- **exp07_modelo_mini** (epoca 10): Every effort moves you of the and, and the--                                   " the--I--I had.
- **exp08_modelo_small** (epoca 10): Every effort moves you, and in the of the the.     "I.     "I, I had been.    "I.  ". "I and I had been, and I his
