# Asphal-LLM: um LLM estilo GPT construído do zero

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Testes](https://img.shields.io/badge/testes-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Baseado em](https://img.shields.io/badge/baseado%20em-LLMs--from--scratch-181717?logo=github)](https://github.com/rasbt/LLMs-from-scratch)

Este repositório contém o código para desenvolver, pré-treinar e ajustar um modelo de
linguagem do tipo GPT, implementado passo a passo, do zero (*From Scratch*), em PyTorch.
Cada componente (tokenização, embeddings, atenção, blocos Transformer, treinamento e
geração de texto) é escrito e testado, capítulo por capítulo, para entender como um LLM
funciona por dentro.

O projeto segue o livro
[*Build a Large Language Model (From Scratch)*](https://www.amazon.com.br/dp/1633437167?lv=shuf&channelId=425&plpRedirect=mhFallback&th=1),
de [Sebastian Raschka](https://github.com/rasbt), e acompanha a estrutura do
[repositório oficial do livro](https://github.com/rasbt/LLMs-from-scratch).

> **Disciplina:** Inteligência Artificial e Sistemas Inteligentes\
> **Professor:** Kleyton Hoffmann\
> **Alunos:** Alan Hoffmann Dos Santos e Jackson

<br>

## Tecnologias

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![tiktoken](https://img.shields.io/badge/tiktoken-0.5%2B-412991)](https://github.com/openai/tiktoken)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab%204-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8%2B-11557C)](https://matplotlib.org/)
[![pytest](https://img.shields.io/badge/pytest-8.0%2B-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)](https://git-scm.com/)

| Tecnologia | Uso no projeto |
|------------|----------------|
| **Python 3.11+** | Linguagem de todo o projeto |
| **PyTorch** | Tensores, camadas do modelo, treinamento e uso de GPU |
| **tiktoken** | Tokenizador BPE do GPT-2 (Capítulo 2 em diante) |
| **Jupyter** | Notebooks de exploração de cada capítulo |
| **Matplotlib** | Gráficos de atenção e curvas de perda do treino |
| **pytest** | Testes automatizados de cada capítulo |

<br>

## Sumário

Cada capítulo tem sua pasta em `sprints/` com o glossário técnico, a análise da Sprint
(`docs/`), os experimentos, os testes e um `README.md` próprio. O código cumulativo da
LLM fica em [`src/`](src/).

> **Dica:** se você é novo no projeto, comece pelo [Capítulo 1](sprints/ch01/) para
> preparar e validar o ambiente antes de rodar os demais.

| Título do capítulo | Código principal (acesso rápido) | Todo o código + material extra |
|--------------------|----------------------------------|--------------------------------|
| Preparação do ambiente | - [requirements.txt](requirements.txt)<br>- [test_ambiente_pytorch.py](sprints/ch01/tests/test_ambiente_pytorch.py) | [./sprints/ch01](sprints/ch01/) |
| ✅ Cap. 1: Entendendo os Large Language Models | - [Glossário do Cap. 1](sprints/ch01/glossario/)<br>- [mapa_conceitual_gpt.md](sprints/ch01/mapa_conceitual_gpt.md) | [./sprints/ch01](sprints/ch01/) |
| ✅ Cap. 2: Trabalhando com dados de texto | - [Notebook do Cap. 2](sprints/ch02/notebooks/InteligenciaArtificial_SistemasInteligentes.ipynb)<br>- [fluxo_sprint2.py](sprints/ch02/experiments/fluxo_sprint2.py) (resumo)<br>- [src/data/](src/data/) | [./sprints/ch02](sprints/ch02/) |
| ✅ Cap. 3: Implementando mecanismos de atenção | - [ch03.ipynb](sprints/ch03/notebooks/ch03.ipynb)<br>- [fluxo_sprint3.py](sprints/ch03/experiments/fluxo_sprint3.py) (resumo)<br>- [src/attention/](src/attention/) | [./sprints/ch03](sprints/ch03/) |
| ✅ Cap. 4: Implementando um modelo GPT do zero | - [ch04.ipynb](sprints/ch04/notebooks/ch04.ipynb)<br>- [fluxo_sprint4.py](sprints/ch04/experiments/fluxo_sprint4.py) (resumo)<br>- [src/model/](src/model/) | [./sprints/ch04](sprints/ch04/) |
| ✅ Cap. 5: Pré-treinamento em dados não rotulados | - [treino_modelo.py](sprints/ch05/experiments/treino_modelo.py)<br>- [estrategias_geracao.py](sprints/ch05/experiments/estrategias_geracao.py)<br>- [src/training/](src/training/) | [./sprints/ch05](sprints/ch05/) |
| ⬜ Cap. 6: Fine-tuning para classificação de texto | Em breve | — |
| ⬜ Cap. 7: Fine-tuning para seguir instruções | Em breve | — |

<br>

O modelo mental abaixo resume o que é construído em cada etapa:

```
Cap. 2  Texto → tokens → IDs → embeddings (+ posição)
Cap. 3  Embeddings → self-attention → atenção causal → multi-head attention
Cap. 4  Multi-head attention + feed-forward + LayerNorm → Transformer Block → GPTModel
Cap. 5  GPTModel + dados → laço de treino → avaliação (loss/perplexity) → geração de texto
Cap. 6-7  Modelo pré-treinado → fine-tuning (classificação / instruções)
```

<br>

## Objetivo

Compreender e implementar os componentes fundamentais de uma arquitetura Transformer,
do processamento de texto até a geração, aplicando o modelo a um corpus do domínio de
**Segurança da Informação** (documentação defensiva, conceitos e terminologia técnica).

O objetivo é didático: entender experimentalmente o funcionamento de cada componente,
não competir com modelos comerciais.

<br>

## Estrutura do repositório

O projeto é organizado por capítulo do livro (`ch0x`), no mesmo padrão de nomenclatura
do repositório oficial. Cada capítulo corresponde a uma Sprint da disciplina (as Sprints
0 e 1 são a mesma entrega). O **código da LLM fica centralizado e cumulativo** em `src/`:
é ali que o modelo cresce, integrando as funcionalidades de cada capítulo.

```
Asphal-LLM/
├── src/                  # código reutilizável e cumulativo da LLM
│   ├── data/             # tokenização, dataset, dataloader, embeddings (Cap. 2)
│   ├── attention/        # self-attention, atenção causal e multi-head (Cap. 3)
│   ├── model/            # GPTModel, Transformer Block, LayerNorm, geração (Cap. 4)
│   └── training/         # perdas, laço de treino, gráficos, checkpoints (Cap. 5)
├── sprints/              # material de cada capítulo/sprint (veja sprints/README.md)
│   ├── ch01/ … ch05/     # glossário, docs, notebooks, experimentos e testes
├── data/                 # corpus de texto compartilhado entre capítulos
├── models/               # checkpoints treinados (não versionados)
├── conftest.py           # torna `src/` importável pelos testes de qualquer capítulo
└── requirements.txt      # dependências do projeto
```

<br>

## Instalação

Requisitos: **Python 3.11+** (desenvolvido e testado em 3.12 e 3.14).

```powershell
# 1. Clonar o repositório
git clone https://github.com/AlanHoff5/Asphal-LLM.git
cd Asphal-LLM

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # Windows (PowerShell)
# source .venv/bin/activate        # Linux / macOS

# 3. Instalar as dependências
pip install -r requirements.txt
```

<br>

## Como rodar

```powershell
# Validar o ambiente (PyTorch + GPU, se houver)
python -m pytest sprints/ch01/tests/ -v -s

# Executar todos os testes do projeto (use -s para ver os prints demonstrativos)
python -m pytest -v

# Fluxos de ponta a ponta de cada capítulo
python sprints/ch02/experiments/fluxo_sprint2.py
python sprints/ch03/experiments/fluxo_sprint3.py
python sprints/ch04/experiments/fluxo_sprint4.py

# Capítulo 5: treinar o GPT e testar as estratégias de geração
python sprints/ch05/experiments/treino_modelo.py
python sprints/ch05/experiments/estrategias_geracao.py
```

Os testes de cada capítulo ficam na respectiva pasta em `sprints/`. O `pytest`, rodado a
partir da raiz, encontra todos automaticamente, e o `conftest.py` torna `src/` importável
em qualquer subpasta.

<br>

## Requisitos de hardware

O código roda em um notebook comum, **sem precisar de GPU**. Se houver uma placa de
vídeo, ela é usada automaticamente.

O teste [`test_ambiente_pytorch.py`](sprints/ch01/tests/test_ambiente_pytorch.py)
detecta NVIDIA (CUDA), AMD (ROCm) ou Apple Silicon (MPS) e, se não houver GPU, cai para
a CPU sem falhar. Quando existe GPU, ele executa operações reais nela (não apenas
`is_available()`), seguindo o mesmo `torch.device` dinâmico que o livro usa a partir do
Capítulo 5. Para conferir qual dispositivo está em uso:

```powershell
python -m src.training.dispositivo
```

<br>

## Referência

Este projeto é um estudo acadêmico baseado no livro e no código de Sebastian Raschka.
Todo o crédito pelo conteúdo original é do autor.

- Livro: [*Build a Large Language Model (From Scratch)*](https://www.amazon.com.br/dp/1633437167?lv=shuf&channelId=425&plpRedirect=mhFallback&th=1), Manning, 2024
- Repositório oficial: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- Autor: [Sebastian Raschka](https://github.com/rasbt)

```bibtex
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```
