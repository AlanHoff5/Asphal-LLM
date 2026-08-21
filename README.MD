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

- Nome 1 — matrícula
- Nome 2 — matrícula

## Estrutura do repositório

```
Asphal-LLM/
├── data/            # corpus (raw = bruto, processed = unificado)
├── src/             # código reutilizável dos componentes do modelo
│   └── data/        # tokenização, dataset e dataloader
├── experiments/     # scripts de experimento por sprint
├── notebooks/       # exploração e estudo
├── docs/            # glossário, relatórios e entregas por sprint
├── models/          # checkpoints (não versionado)
├── requirements.txt # dependências (geradas com pip freeze)
└── README.md
```

## Como rodar

Requisitos: Python 3.11.

```powershell
# 1. Criar e ativar o ambiente virtual
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source venv/bin/activate         # Linux / macOS

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Executar um componente (exemplo: tokenizador)
python src/data/tokenizer.py
```

## Progresso por Sprint

| Sprint | Capítulo | Conteúdo | Status |
|--------|----------|----------|--------|
| 1 | Cap. 1 | Introdução aos LLMs, glossário, mapa conceitual | ✅ |
| 2 | Cap. 2 | Tokenização, embeddings, positional embeddings, DataLoader | 🚧 |
| 3 | Cap. 3 | Mecanismos de atenção (self / causal / multi-head) | ⬜ |
| 4 | Cap. 4 | Arquitetura GPT, Transformer Block | ⬜ |
| 5 | Cap. 5 | Treinamento e geração de texto | ⬜ |
| 6 | Caps. 6-7 | Fine-tuning | ⬜ |

## Referência

RASCHKA, Sebastian. *Build a Large Language Model (From Scratch)*. Manning, 2024.