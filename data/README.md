# data/

## `the-verdict.txt`

Corpus público de exemplo usado no Capítulo 2 do livro (domínio público),
usado nos testes e no notebook para reproduzir exatamente os experimentos
do livro antes de aplicar o pipeline ao corpus do projeto.

## `comptia_security_pluse_701/cleaned_data.txt` (pendente)

Corpus próprio do projeto, do domínio de Segurança da Informação (material
de estudo para a certificação CompTIA Security+), usado para os testes
`test_amostra_do_corpus_real_comptia` (`sprints/ch02/tests/`) e citado em
`sprints/ch02/docs/sprint2_analise.md` (682.480 caracteres, 110.334 tokens,
vocabulário de 9.291 entradas na tokenização por palavras).

Esse arquivo **não está no repositório**. Ele tinha sido adicionado como
submódulo Git (provavelmente porque a pasta continha um `.git` próprio no
momento do `git add`), mas o submódulo nunca teve um `.gitmodules`
correspondente nem foi enviado a lugar nenhum — ou seja, mesmo no primeiro
commit em que aparece, ele já estava vazio e quebrado. Não há como
recuperá-lo pelo histórico do Git porque o conteúdo real nunca chegou a ser
commitado neste repositório.

Para corrigir: coloque o arquivo de texto real em
`data/comptia_security_pluse_701/cleaned_data.txt` (texto puro, UTF-8) e dê
`git add` normalmente — sem inicializar um repositório Git dentro dessa
pasta. Assim que o arquivo existir:

- os dois testes que hoje aparecem como `SKIPPED` passam a rodar de verdade;
- `sprints/ch02/experiments/fluxo_sprint2.py` passa a usar o corpus real
  automaticamente em vez do texto de exemplo (ver `CORPUS_COMPTIA` no topo
  do arquivo).
