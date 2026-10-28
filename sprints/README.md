# Sprints

Cada capitulo do livro *Build a Large Language Model (From Scratch)* tem sua
propria pasta aqui, nomeada por capitulo do livro (`ch0x`), no mesmo padrao 
usado no repositorio oficial do Sebastian Raschka.
Sprint 0 e 1, são a mesma entrega conforme a trilha do professor.

Cada pasta guarda o material especifico daquele capitulo/sprint: glossario,
documentacao/relatorios, notebooks de exploracao, experimentos e testes.

O **codigo reutilizavel do modelo** (tokenizador, dataset, dataloader,
embeddings, e o que vier nos proximos capitulos) fica centralizado em
`src/`, fora das pastas de capitulo. `src/`: representa sempre
a versao mais atual e integrada da LLM, somando o que foi construido em
todos os capitulos ate aqui. As pastas `ch0x/` guardam o material de estudo
e experimentacao daquela etapa, nao uma copia do codigo do modelo.

## Convencao de pastas por capitulo

```
sprints/ch0x/
├── README.md          # resumo, status e entregaveis do capitulo/sprint
├── glossario/         # glossario tecnico do capitulo (ex.: ch0x.md)
├── docs/              # relatorios, analises e respostas (quando houver)
├── notebooks/         # exploracao/estudo em Jupyter (quando houver)
├── experiments/       # scripts que demonstram o fluxo do capitulo (quando houver)
└── tests/             # testes automatizados especificos do capitulo
```

## Indice

| Capitulo/Sprint | Conteudo | Pasta | Status |
|------------------|----------|-------|--------|
| 1 (Sprints 0+1) | Preparacao do ambiente + Introducao aos LLMs | [`ch01/`](ch01/) | ✅ |
| 2 (Sprint 2) | Trabalhando com Dados Textuais | [`ch02/`](ch02/) | ✅ |
| 3 (Sprint 3) | Mecanismos de Atencao | [`ch03/`](ch03/) | ✅ |
| 4 (Sprint 4) | Arquitetura GPT | *(ainda nao criada)* | ⬜ |
| 5 (Sprint 5) | Treinamento e geracao de texto | *(ainda nao criada)* | ⬜ |
| 6-7 (Sprint 6) | Fine-tuning | *(ainda nao criada)* | ⬜ |