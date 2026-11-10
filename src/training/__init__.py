# Chapter 5 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - treinamento do modelo (pretraining)
#
# Modulos:
#   dispositivo.py  escolhe onde rodar: GPU (CUDA) ou CPU
#   dados.py        divide o texto em treino/validacao e cria os DataLoaders
#   losses.py       cross-entropy por lote e por loader, perplexity
