# Asphal-LLM

Implementação incremental de um modelo de linguagem (LLM) do tipo GPT, construído
do 0 com base na obra *Build a Large Language Model (From Scratch)*, de
Sebastian Raschka.

Projeto da disciplina **Inteligência Artificial e Sistemas Inteligentes**, professor
Kleyton Hoffmann.

## Objetivo

Compreender e implementar os componentes fundamentais de uma arquitetura Transformer,
do processamento de texto até a geração, aplicando o modelo a um corpus do domínio de
**Segurança da Informação** (documentação defensiva, conceitos e terminologia técnica).

O objetivo é didático: entender experimentalmente o funcionamento de cada componente,
não competir com modelos comerciais.

## Integrantes

- Alan Hoffmann Dos Santos
- Jackson

## Estrutura do repositório

O projeto é organizado por capítulo do livro (`ch0x`), no mesmo
padrão de nomenclatura usado no [repositório oficial do Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch).
Cada capítulo corresponde a uma Sprint da trilha, conforme a trilha Sprint 0 e 1 são a mesma entrega.
O **código da LLM fica centralizado e cumulativo** em
`src/`, ou seja, é onde ficará o corpo principal da nossa LLM com todas as funcionalidades de cada capitulo integradas.

```
Asphal-LLM/
├── src/                  # código reutilizável e cumulativo da LLM (todos os capítulos)
│   └── data/              # tokenização, dataset, dataloader, embeddings (Cap. 2+)
├── sprints/              # material de cada capítulo/sprint: veja sprints/README.md
│   ├── ch01/               # Sprints 0+1: ambiente, teste de PyTorch/GPU, glossário do Cap. 1
│   ├── ch02/               # Sprint 2: glossário, docs, notebook, experimentos e testes do Cap. 2
│   └── ch03/               # Sprint 3: glossário do Cap. 3 
├── data/                 # corpus bruto compartilhado entre capítulos
├── conftest.py           # faz `src/` ser importável pelos testes de qualquer capítulo
├── models/               # checkpoints treinados 
├── requirements.txt      # dependências 
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

# 3. Validar o ambiente (PyTorch + GPU, se houver)
python -m pytest sprints/ch01/tests/ -v -s

# 4. Executar todos os testes automatizados do projeto
python -m pytest -v

# 5. Executar os testes com os prints demonstrativos
python -m pytest -s -v

# 6. Demonstrar o fluxo completo do Capítulo 2, de ponta a ponta
python sprints/ch02/experiments/fluxo_sprint2.py
```

### GPU / placa de vídeo

O teste em [`sprints/ch01/tests/test_ambiente_pytorch.py`](sprints/ch01/tests/test_ambiente_pytorch.py)
confirma que o PyTorch consegue rodar na placa de vídeo disponível. Ele é
agnóstico ao fabricante: detecta automaticamente NVIDIA (CUDA), AMD (ROCm) ou
Apple Silicon (MPS) e, se a máquina não tiver GPU, cai para CPU sem falhar —
funciona em qualquer computador do grupo. Quando existe GPU, o teste roda
operações reais nela (não só checa `is_available()`), seguindo o mesmo padrão
de `torch.device` dinâmico que o livro usa a partir do Capítulo 5.

Os testes específicos de cada capítulo ficam dentro da respectiva pasta em
`sprints/`. `pytest` (rodado a partir da raiz do repositório) os descobre
automaticamente, pois `conftest.py` cuida de tornar `src/` importável em
qualquer subpasta.

## Progresso por Sprint / Capítulo

| Capítulo | Sprint | Conteúdo | Status | Pasta |
|----------|--------|----------|--------|-------|
| 1 | 0 + 1 | Preparação do ambiente, teste de PyTorch/GPU, introdução aos LLMs | ✅ | [`sprints/ch01/`](sprints/ch01/) |
| 2 | 2 | Tokenização, embeddings, positional embeddings, DataLoader | ✅ | [`sprints/ch02/`](sprints/ch02/) |
| 3 | 3 | Mecanismos de atenção (self / causal / multi-head) | ✅ | [`sprints/ch03/`](sprints/ch03/) |
| 4 | 4 | Arquitetura GPT, Transformer Block | ⬜ | — |
| 5 | 5 | Treinamento e geração de texto | ⬜ | — |
| 6-7 | 6 | Fine-tuning | ⬜ | — |

## Referência

RASCHKA, Sebastian. *Build a Large Language Model (From Scratch)*. Manning, 2024.
