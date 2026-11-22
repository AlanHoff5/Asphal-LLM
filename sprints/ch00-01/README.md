# Capítulo 1 - Introdução aos LLMs (Sprint 0 + Sprint 1)

Status: ✅ concluída

Na trilha, a Sprint 0 (preparação do ambiente) e a Sprint 1 (Capítulo 1 do
livro) são a mesma entrega, então ficam juntas nesta pasta, seguindo o
Capítulo 1 - *Understanding Large Language Models* - da obra de Sebastian
Raschka.

## O que foi feito

- Repositório do projeto criado no GitHub.
- Ambiente Python + PyTorch configurado e validado.
- Leitura do Capítulo 1 e do apêndice de PyTorch do livro: o que caracteriza
  um LLM, diferença entre IA/Machine Learning/Deep Learning, organização dos
  modelos GPT e o fluxo geral de treinamento.
- Glossário técnico do Capítulo 1.
- Mapa conceitual da arquitetura GPT.

## Conteúdo desta pasta

- [`glossario/ch01.md`](glossario/ch01.md) - glossário técnico cumulativo do
  Capítulo 1.
- [`mapa_conceitual_gpt.md`](mapa_conceitual_gpt.md) - mapa conceitual da
  arquitetura GPT, ligando os termos do glossário ao pipeline completo do
  projeto (Texto → Tokenização → Embeddings → Attention → Transformer →
  GPT → Treinamento → Geração).
- [`tests/test_ambiente_pytorch.py`](tests/test_ambiente_pytorch.py) - teste
  que valida o ambiente PyTorch e a GPU (ver abaixo).

Não há código de implementação do modelo nesta etapa - a primeira
implementação do projeto começa no [Capítulo 2](../ch02/).

## Teste do ambiente (PyTorch + GPU)

O professor apontou que o repositório não tinha nenhum teste validando o
uso de GPU. Isso foi corrigido em [`tests/test_ambiente_pytorch.py`](tests/test_ambiente_pytorch.py):

```powershell
python -m pytest sprints/ch01/tests/test_ambiente_pytorch.py -v -s
```

O teste detecta automaticamente a melhor placa de vídeo disponível (NVIDIA
CUDA, AMD ROCm ou Apple Silicon/MPS) e, se nenhuma existir, cai para CPU sem
falhar. Quando há GPU, ele executa operações reais nela (não só verifica
`is_available()`), provando que o ambiente realmente consegue treinar na
placa de vídeo - o mesmo padrão de `torch.device` dinâmico usado pelo livro
a partir do Capítulo 5.

## Entregáveis

- [x] Link do repositório do projeto.
- [x] Ambiente computacional configurado e funcional.
- [x] `README.md` com a descrição do projeto (raiz do repositório).
- [x] Glossário do Capítulo 1.
- [x] Mapa conceitual da arquitetura GPT.
- [x] Teste automatizado do ambiente PyTorch/GPU.
