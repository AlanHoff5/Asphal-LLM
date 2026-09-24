# Capítulo 3 - Mecanismos de Atenção (Sprint 3)

Status: ✅ implementação, experimentos e análise concluídos (falta o Quiz 3, individual, fora do escopo do repositório)

Capítulo 3 do livro (*Coding Attention Mechanisms*): Self-Attention,
Scaled Dot-Product Attention, Causal Attention e Multi-Head Attention,
a partir dos dados preparados no [Capítulo 2](../ch02/).

## Conteúdo desta pasta

```
ch03/
├── glossario/ch03.md                  # glossario tecnico do Capitulo 3
├── docs/
│   └── sprint3_analise.md             # analise tecnica dos experimentos
├── notebooks/
│   └── ch03.ipynb                     # notebook executado ponta a ponta (secoes 3.3 a 3.6)
├── experiments/
│   ├── fluxo_sprint3.py               # demonstracao + 7 experimentos + visualizacao
│   └── resultados/matriz_de_atencao.png
└── tests/
    ├── test_self_attention.py         # secoes 3.3 e 3.4
    ├── test_causal_attention.py       # secao 3.5
    └── test_multi_head_attention.py   # secao 3.6
```

O **código de implementação** (Self-Attention, Causal Attention, Multi-Head
Attention) fica em [`src/attention/`](../../src/attention/), pelo mesmo
motivo do Capítulo 2: é reutilizável e será a base do `Transformer Block`
na Sprint 4.

## Como rodar

```powershell
# Testes automatizados da Sprint 3
python -m pytest sprints/ch03/tests/ -v

# Testes com os prints demonstrativos
python -m pytest sprints/ch03/tests/ -s -v

# Fluxo completo + 7 experimentos + visualizacao da matriz de atencao
python sprints/ch03/experiments/fluxo_sprint3.py
```

O script de experimentos salva um heatmap por cabeça de atenção em
`experiments/resultados/matriz_de_atencao.png` (requer `matplotlib`,
incluído em `requirements.txt`).

## O que foi implementado

- **Self-Attention sem pesos treináveis** (`attention_scores_simples`) -
  validada contra o valor de referência do livro.
- **Self-Attention com pesos treináveis**, `SelfAttention_v1` (nn.Parameter,
  fiel ao livro) e `SelfAttention_v2` (nn.Linear, usada dali em diante),
  ambas com Scaled Dot-Product Attention.
- **Causal Attention**, em lote, com máscara causal (`torch.triu` +
  `masked_fill_`) e Dropout sobre os pesos de atenção.
- **Multi-Head Attention**, nas duas versões do livro:
  `MultiHeadAttentionWrapper` (didática, várias cabeças em loop) e
  `MultiHeadAttention` (eficiente, uma única projeção dividida em cabeças
  via `view`/`transpose`) - esta última é a que a Sprint 4 vai reaproveitar.

Detalhes técnicos, respostas conceituais e os resultados numéricos de cada
experimento estão em [`docs/sprint3_analise.md`](docs/sprint3_analise.md).

O notebook [`notebooks/ch03.ipynb`](notebooks/ch03.ipynb) percorre o mesmo
conteúdo de forma narrativa (célula a célula, com os resultados já
executados e o heatmap inline), no mesmo formato do notebook do
[Capítulo 2](../ch02/notebooks/) - e do próprio repositório do Sebastian
Raschka, que também tem um notebook principal por capítulo
(`ch03/01_main-chapter-code/ch03.ipynb`).


