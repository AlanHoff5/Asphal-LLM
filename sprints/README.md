# Sprints

Cada etapa da trilha *Build a Large Language Model (From Scratch)* tem sua
propria pasta aqui, nomeada por capitulo do livro (`ch01`, `ch02`, `ch03`,
...), no mesmo padrao usado no repositorio oficial do Sebastian Raschka
(https://github.com/rasbt/LLMs-from-scratch). A Sprint 0 (preparacao do
ambiente) e a Sprint 1 (Capitulo 1) sao a mesma entrega no curso, entao
ficam juntas em `ch01/`.

Cada pasta guarda o material especifico daquele capitulo/sprint: glossario,
documentacao/relatorios, notebooks de exploracao, experimentos e testes.

O **codigo reutilizavel do modelo** (tokenizador, dataset, dataloader,
embeddings, e o que vier nos proximos capitulos) fica centralizado em
`src/`, fora das pastas de capitulo. `src/` e cumulativo: representa sempre
a versao mais atual e integrada da LLM, somando o que foi construido em
todos os capitulos ate aqui. As pastas `chNN/` guardam o material de estudo
e experimentacao daquela etapa, nao uma copia do codigo do modelo.

## Convencao de pastas por capitulo

```
sprints/chNN/
├── README.md        # resumo, status e entregaveis do capitulo/sprint
├── glossario/         # glossario tecnico do capitulo (ex.: chNN.md)
├── docs/              # relatorios, analises e respostas (quando houver)
├── notebooks/         # exploracao/estudo em Jupyter (quando houver)
├── experiments/       # scripts que demonstram o fluxo do capitulo (quando houver)
└── tests/             # testes automatizados especificos do capitulo
```

Nem todo capitulo usa todas as subpastas - crie apenas as que fizerem
sentido para o conteudo daquela etapa. O arquivo de glossario segue o nome
do capitulo (`ch01.md`, `ch02.md`, `ch03.md`, ...), igual ao padrao de
nomenclatura dos notebooks no repositorio do livro.

## Indice

| Capitulo/Sprint | Conteudo | Pasta | Status |
|------------------|----------|-------|--------|
| 1 (Sprints 0+1) | Preparacao do ambiente + Introducao aos LLMs | [`ch01/`](ch01/) | ✅ |
| 2 (Sprint 2) | Trabalhando com Dados Textuais | [`ch02/`](ch02/) | ✅ |
| 3 (Sprint 3) | Mecanismos de Atencao | [`ch03/`](ch03/) | ✅ |
| 4 (Sprint 4) | Arquitetura GPT | *(ainda nao criada)* | ⬜ |
| 5 (Sprint 5) | Treinamento e geracao de texto | *(ainda nao criada)* | ⬜ |
| 6-7 (Sprint 6) | Fine-tuning | *(ainda nao criada)* | ⬜ |

Quando um novo capitulo comecar, crie a pasta `sprints/chNN/` seguindo a
convencao acima e adicione uma linha nesta tabela.
