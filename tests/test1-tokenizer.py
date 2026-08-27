# Chapter 2 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Alan Hoffmann Dos Santos
# Teste de Tokenizador

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.data.tokenizer as tokenizer


# Testando a função tokenize
texto = "Olá, mundo! Este é um teste de tokenização."
print("Texto original:", texto)
tokens = tokenizer.tokenize(texto)
print("Tokens:", tokens)

# Testando a classe SimpleTokenizerV1
vocab = {"Olá": 0, ",": 1, "mundo": 2, "!": 3, "Este": 4, "é": 5, "um": 6, "teste": 7, "de": 8, "tokenização": 9, ".": 10}
tokenizer_v1 = tokenizer.SimpleTokenizerV1(vocab)
print("\nTestando SimpleTokenizerV1")
texto = "Olá, mundo! Este é um teste de tokenização."
# Codificando o texto em tokens
tokens_v1 = tokenizer_v1.encode(texto)
print("Tokens codificados:", tokens_v1)
# Decodificando os tokens de volta para texto
decoded_text_v1 = tokenizer_v1.decode(tokens_v1)
print("Texto decodificado:", decoded_text_v1)
