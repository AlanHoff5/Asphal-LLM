# Mapa Conceitual da Arquitetura GPT

Entregável da Sprint 1 (cronograma oficial, 06/08): "elaboração do glossário,
Quiz 1 e **mapa conceitual da arquitetura GPT**". Este documento organiza
visualmente os termos definidos em [`glossario/ch01.md`](glossario/ch01.md)
no fluxo geral que será implementado ao longo do projeto, do texto bruto até
a geração de texto — o mesmo roteiro descrito na Seção 9 ("Resultado Final
Esperado") do documento do Projeto LLM.

## 1. Visão geral do pipeline

```mermaid
flowchart TD
    A["Texto bruto\n(Raw Text)"] --> B["Tokenização\n(Tokenizer)"]
    B --> C["Token IDs\n(Vocabulary)"]
    C --> D["Token Embedding"]
    C --> E["Positional Embedding"]
    D --> F["Input Embedding\n(Token + Position)"]
    E --> F
    F --> G["Multi-Head Attention"]
    G --> H["Transformer Block\n(Attention + Feed Forward + Layer Norm + Residual)"]
    H --> H
    H --> I["Modelo GPT\n(Decoder-Only, N blocos empilhados)"]
    I --> J["Pretraining\n(Next-Word Prediction sobre Raw Text)"]
    J --> K["Foundation Model"]
    K --> L["Fine-Tuning\n(Instruction ou Classification)"]
    L --> M["Downstream Task"]
    I --> N["Geração de Texto\n(Autoregressive Model)"]

    style A fill:#eee,stroke:#333
    style B fill:#cde4ff,stroke:#333
    style C fill:#cde4ff,stroke:#333
    style D fill:#cde4ff,stroke:#333
    style E fill:#cde4ff,stroke:#333
    style F fill:#cde4ff,stroke:#333
    style G fill:#ffe4b3,stroke:#333
    style H fill:#ffe4b3,stroke:#333
    style I fill:#ffe4b3,stroke:#333
    style J fill:#d9f2d9,stroke:#333
    style K fill:#d9f2d9,stroke:#333
    style L fill:#d9f2d9,stroke:#333
    style M fill:#d9f2d9,stroke:#333
    style N fill:#d9f2d9,stroke:#333
```

Azul = Capítulo 2 (concluído, `src/data/`). Laranja = Capítulo 3 em diante
(atenção e arquitetura GPT, em andamento). Verde = treinamento, fine-tuning
e geração de texto (Capítulos 5 a 7, ainda não iniciados).

## 2. Onde cada Sprint entra no mapa

| Etapa do mapa | Conceitos (glossário) | Sprint | Status |
|---|---|---|---|
| Texto bruto → Token IDs | `Tokenization`, `Token`, `Vocabulary`, `Token ID`, `Special Context Tokens`, `Byte Pair Encoding` | 2 | ✅ |
| Token/Positional Embedding | `Token Embedding`, `Positional Embedding`, `Input Embedding` | 2 | ✅ |
| Multi-Head Attention | `Attention`, `Self-Attention`, `Query/Key/Value`, `Causal Attention`, `Multi-Head Attention` | 3 | ⬜ em andamento |
| Transformer Block / GPT | `Transformer`, `Decoder`, `Decoder-Only Architecture`, `GPT` | 4 | ⬜ |
| Pretraining / Foundation Model | `Pretraining`, `Next-Word Prediction`, `Self-Supervised Learning`, `Foundation Model`, `Autoregressive Model` | 5 | ⬜ |
| Fine-Tuning / Downstream Task | `Fine-Tuning`, `Instruction Fine-Tuning`, `Classification Fine-Tuning`, `Downstream Task` | 6 | ⬜ |

## 3. Por que Decoder-Only (GPT) e não Encoder (BERT)

O Transformer original tem `Encoder` e `Decoder`. O GPT descarta o
`Encoder` e empilha só `Decoder`s (`Decoder-Only Architecture`), treinado
por `Next-Word Prediction` — geração token a token, da esquerda para a
direita. O `BERT` é o caminho oposto: usa só o `Encoder`, treinado por
`Masked Word Prediction`, e por isso é melhor em classificação do que em
geração. Essa escolha arquitetural é o motivo de o mecanismo de atenção que
vamos implementar na Sprint 3 precisar ser **causal** (`Causal Attention`):
cada posição só pode olhar para tokens anteriores a ela, nunca para o
futuro — é isso que torna a geração autoregressiva possível.

## 4. Hierarquia de escala (IA → LLM)

```mermaid
flowchart LR
    IA["Inteligência Artificial"] --> ML["Machine Learning"]
    ML --> DL["Deep Learning\n(PyTorch)"]
    DL --> LLM["Large Language Model\n(GPT, Decoder-Only)"]
```

Cada seta representa uma especialização: `Deep Learning` é o subcampo do
`Machine Learning` baseado em redes de múltiplas camadas ocultas; um `LLM`
é uma aplicação de `Deep Learning` a texto, usando `Transformer` como
espinha dorsal e `Embedding` como porta de entrada — a mesma cadeia descrita
no glossário do Capítulo 1.
