import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import build_vocab, create_tokenizer, tokenize


# 3.1: o texto foi separado em tokens; a pontuacao virou um token proprio.
texto = "Firewall bloqueia ataques."

tokens = tokenize(texto)

print("Texto:", texto)
print("Tokens:", tokens)
print("Quantidade de tokens:", len(tokens))

# 3.2: o vocabulario registra cada token uma unica vez e associa um ID inteiro.
vocabulario = build_vocab(texto)

print("\nVocabulário:")
print(vocabulario)
print("Tamanho do vocabulário:", len(vocabulario))

# O mesmo vocabulario permite codificar a frase e depois reconstruir o texto.
tokenizador = create_tokenizer(texto)
token_ids = tokenizador.encode(texto)
texto_recuperado = tokenizador.decode(token_ids)

print("\nToken IDs:")
print(token_ids)
print("Texto recuperado:")
print(texto_recuperado)

# Teste de palavra desconhecida: a versao 2 usa <|unk|> para um token
# que nao apareceu no texto usado na criacao do vocabulario.
texto_novo = "Firewall investiga ataques."
ids_novos = tokenizador.encode(texto_novo)

print("\nTeste de palavra desconhecida:")
print("Texto novo:", texto_novo)
print("Token IDs:", ids_novos)
print("Texto decodificado:", tokenizador.decode(ids_novos))