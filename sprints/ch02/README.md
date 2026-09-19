# Capítulo 2 - Trabalhando com Dados Textuais (Sprint 2)

Status: ✅ concluida

Capitulo 2 do livro (*Working with Text Data*): tokenizacao, vocabulario,
Token IDs, preparacao de sequencias de treinamento, embeddings, positional
embeddings e DataLoader.

## Conteudo desta pasta

```
ch02/
├── glossario/ch02.md   # glossario tecnico do Capitulo 2
├── docs/
│   ├── sprint2_guia_de_estudo.md            # guia de estudo do capitulo
│   ├── sprint2_respostas.md                 # respostas as questoes da sprint
│   └── sprint2_analise.md                   # analise critica dos resultados
├── notebooks/
│   └── InteligenciaArtificial_SistemasInteligentes.ipynb  # exploracao/estudo
└── experiments/
    └── fluxo_sprint2.py                     # demonstracao reproduzivel do fluxo completo
```

O **codigo de implementacao** (tokenizador, dataset, dataloader, embeddings)
fica em [`src/data/`](../../src/data/), pois e reutilizado pelas proximas
sprints. Os **testes automatizados** desta sprint ficam em
[`tests/`](tests/), ao lado deste README.

## Como rodar

```powershell
# Testes automatizados da Sprint 2
python -m pytest sprints/ch02/tests/ -v

# Testes com os prints demonstrativos
python -m pytest sprints/ch02/tests/ -s -v

# Fluxo completo de ponta a ponta
python sprints/ch02/experiments/fluxo_sprint2.py

# Notebook de exploracao (abra a partir desta pasta para o
# download automatico de the-verdict.txt funcionar corretamente)
jupyter lab sprints/ch02/notebooks/InteligenciaArtificial_SistemasInteligentes.ipynb
```

## Entregaveis

- [x] Glossario atualizado do Capitulo 2.
- [x] Codigo de tokenizacao (`src/data/tokenizer.py`).
- [x] Vocabulario e Token IDs (`src/data/tokenizer.py`).
- [x] Preparacao das sequencias de treinamento (`src/data/dataset.py`).
- [x] Embeddings e Positional Embeddings (`src/data/embeddings.py`).
- [x] Estrutura de carregamento dos dados (`src/data/dataloader.py`).
- [x] Experimentos realizados (`experiments/fluxo_sprint2.py`, notebook).
- [x] Analise dos resultados (`docs/sprint2_analise.md`).
- [x] Codigo organizado no repositorio.
