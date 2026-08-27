# Asphal-LLM

Implementação incremental de um modelo de linguagem (LLM) do tipo GPT, construído
"from scratch" com base na obra *Build a Large Language Model (From Scratch)*, de
Sebastian Raschka.

Projeto da disciplina **Inteligência Artificial e Sistemas Inteligentes**.

## Objetivo

Compreender e implementar os componentes fundamentais de uma arquitetura Transformer,
do processamento de texto até a geração, aplicando o modelo a um corpus do domínio de
**Segurança da Informação** (documentação defensiva, conceitos e terminologia técnica).

O objetivo é didático: entender experimentalmente o funcionamento de cada componente,
não competir com modelos comerciais.

## Integrantes

- Alan Hoffmann
- Jackson

## Estrutura do repositório

```
Asphal-LLM/
├── data/            # corpus (raw = bruto, processed = unificado)
├── src/             # código reutilizável dos componentes do modelo
│   └── data/        # tokenização, dataset e dataloader
├── experiments/     # demonstrações reproduzíveis por sprint
├── notebooks/       # exploração e estudo
├── docs/            # glossário, relatórios e respostas por sprint
├── models/          # checkpoints (não versionado)
├── requirements.txt # dependências (geradas com pip freeze)
└── README.MD
```

## Como rodar

Requisitos: Python 3.11+ (desenvolvido e testado em 3.12 e 3.14).

```powershell
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # Windows (PowerShell)
# source .venv/bin/activate        # Linux / macOS

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Executar todos os testes automatizados da Sprint 2
python -m pytest -v

# 4. Executar os testes com os prints demonstrativos
python -m pytest tests/ -s -v

# 5. Demonstrar o fluxo completo da Sprint 2 de ponta a ponta
python experiments/fluxo_sprint2.py
```

Os testes cobrem: tokenização e vocabulário (`test_tokenizer_sprint2.py`),
pares entrada-alvo (`test_dataset_sprint2.py`), embeddings e o efeito da
posição (`test_embeddings.py`) e o DataLoader (`test_dataloader.py`).

Os experimentos exploratórios da Sprint 2 estão no notebook
`notebooks/InteligenciaArtificial_SistemasInteligentes.ipynb`. A demonstração
reproduzível do fluxo completo está em `experiments/fluxo_sprint2.py`. A
documentação das respostas técnicas está em `docs/sprint2_respostas.md`, e a
análise dos resultados está em `docs/sprint2_analise.md`.

## Progresso por Sprint

| Sprint | Capítulo | Conteúdo | Status |
|--------|----------|----------|--------|
| 1 | Cap. 1 | Introdução aos LLMs, glossário, mapa conceitual | ✅ |
| 2 | Cap. 2 | Tokenização, embeddings, positional embeddings, DataLoader | ✅ |
| 3 | Cap. 3 | Mecanismos de atenção (self / causal / multi-head) | ⬜ |
| 4 | Cap. 4 | Arquitetura GPT, Transformer Block | ⬜ |
| 5 | Cap. 5 | Treinamento e geração de texto | ⬜ |
| 6 | Caps. 6-7 | Fine-tuning | ⬜ |

## Referência

RASCHKA, Sebastian. *Build a Large Language Model (From Scratch)*. Manning, 2024.
