# Capítulo 5 - Pré-treinamento em dados não rotulados (Sprint 5)

Status: ✅ implementação, experimentos e análise concluídos

Capítulo 5 do livro (*Pretraining on Unlabeled Data*): avaliação do modelo
com cross-entropy e perplexity, laço de treinamento, experimentos de
treino, estratégias de geração (temperatura e top-k) e salvamento e
carregamento dos pesos. Ponto de partida: o GPT do [Capítulo 4](../ch04/).

## Conteúdo desta pasta

```
ch05/
├── glossario/ch05.md                    # glossario tecnico do Capitulo 5
├── docs/
│   └── sprint5_analise.md               # analise tecnica (Entregaveis 2 a 7)
├── experiments/
│   ├── avaliacao_modelo.py              # Entregavel 2: cross-entropy e perplexity passo a passo
│   ├── treino_modelo.py                 # Entregavel 3: treino do GPT (livro ou configuravel)
│   ├── experimentos_treino.py           # Entregavel 4: fila de 8 experimentos + resumo
│   ├── estrategias_geracao.py           # Entregavel 5: gulosa x temperatura x top-k
│   ├── salvar_carregar.py               # Entregavel 6: carregar, conferir e continuar o treino
│   └── resultados/                      # JSON e resumos (.md) de cada script; .png gerados localmente
└── tests/
    ├── test_losses.py                   # secoes 5.1.2 e 5.1.3
    ├── test_treino.py                   # secao 5.2 e apendice D
    └── test_checkpoint.py               # secao 5.4
```

O **código de implementação** fica em [`src/training/`](../../src/training/):
`dispositivo.py`, `dados.py`, `losses.py`, `treino.py`, `graficos.py` e
`checkpoint.py`. Ele reaproveita, sem alterações, o `create_dataloader_v1`
(Sprint 2), o `GPTModel` e o `generate` (Sprint 4).

## Como rodar

Use o ambiente do projeto com PyTorch CUDA para treinar na GPU
(`python -m src.training.dispositivo` mostra se a GPU está sendo usada).

```powershell
# Testes automatizados da Sprint 5
python -m pytest sprints/ch05/tests/ -v

# Entregavel 2 - avaliacao do modelo sem treino (< 1 min)
python sprints/ch05/experiments/avaliacao_modelo.py

# Entregavel 3 - treino do livro: GPT-2 124M, 10 epocas (~1 min na GTX 1650)
python sprints/ch05/experiments/treino_modelo.py
#   outras configuracoes: --config mini|small, --epocas, --lr, --batch-size,
#   --drop-rate, --weight-decay, --warmup 0.2 --grad-clip 1.0, --nome

# Entregavel 4 - os 8 experimentos em sequencia (~13 min) e o resumo comparativo
python sprints/ch05/experiments/experimentos_treino.py

# Entregaveis 5 e 6 - usam o modelo salvo pelo treino_modelo.py
python sprints/ch05/experiments/estrategias_geracao.py
python sprints/ch05/experiments/salvar_carregar.py
```

Os pesos são salvos em `models/ch05/` (fora do git: cerca de 1,9 GB com o
otimizador). Os gráficos `.png` também são gerados localmente em
`experiments/resultados/`.

## Principais resultados

| | Resultado |
|---|---|
| Perda inicial (sem treino) | treino 10,988 · validação 10,981 (igual ao livro; chute uniforme = 10,825) |
| Treino do livro, 10 épocas | treino **0,420** · validação **6,452** (livro: 0,391 · 6,452), em 63 s na GTX 1650 |
| Overfitting | menor validação **6,087** na época 5; com 30 épocas: treino 0,006 e validação 7,297 |
| Taxa de aprendizado | lr 0,001 não converge (val. 6,82); com warmup + cosine o mesmo lr chega a 6,23 |
| Geração | top-k 25 + T 1,4 = mais diverso sem perder sentido; sem treino, nenhuma estratégia gera texto coerente |
| Salvar e carregar | perda e texto idênticos após carregar; pesos 623 MB, com AdamW 1.862 MB |

Análise completa, tabelas e discussão em
[`docs/sprint5_analise.md`](docs/sprint5_analise.md).
