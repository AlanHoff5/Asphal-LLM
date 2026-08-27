# Teste do Dataset - item 3.3 da Sprint 2

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GPTDatasetV1
from src.data.tokenizer import create_tokenizer, load_text, tokenize


# O texto e convertido em Token IDs antes de virar uma amostra do Dataset.
texto = "um dois tres quatro cinco seis"
tokenizador = create_tokenizer(texto)
tokens = tokenize(texto)
token_ids = tokenizador.encode(texto)


print("TESTE DO DATASET - 3.3")
print("\nTEXTO E TOKEN IDS")
print("Texto:", texto)
print("Tokens:", tokens)
print("Token IDs:", token_ids)

# max_length define quantos tokens cada entrada tera; stride define o avanco
# da janela sobre a sequencia de Token IDs.
max_length = 3 # Cada entrada tem 3 tokens
stride = 1 # Uma palavra por vez
dataset = GPTDatasetV1(texto, tokenizador, max_length, stride) # Cria o Dataset com janelas 
# de entrada e alvo


print("\nCONFIGURACAO DO DATASET")
print("Tamanho do contexto (max_length):", max_length)
print("Avanco da janela (stride):", stride) # Quantas tokens a janela avanca a cada amostra
print("Quantidade de amostras:", len(dataset))


# Cada amostra possui uma entrada e um alvo. O alvo comeca no Token ID seguinte
# e ensina o modelo a prever o proximo token.
print("\nJANELAS DE ENTRADA E ALVO")
for indice in range(len(dataset)):
    entrada, alvo = dataset[indice]
    print(f"Amostra {indice}:")
    print("  Entrada:", entrada.tolist())
    print("  Alvo:   ", alvo.tolist())

# O PyTorch precisa receber tensores longos para usa-los como indices em
# camadas de embedding.
entrada, alvo = dataset[0]
print("\nTIPOS E DIMENSOES")
print("Tipo da entrada:", entrada.dtype)
print("Tipo do alvo:", alvo.dtype)
print("Shape da entrada:", tuple(entrada.shape))
print("Shape do alvo:", tuple(alvo.shape))

# Verificacoes principais do item 3.3.
assert isinstance(entrada, torch.Tensor)
assert isinstance(alvo, torch.Tensor)
assert entrada.dtype == torch.long
assert alvo.dtype == torch.long
assert tuple(entrada.shape) == (max_length,)
assert tuple(alvo.shape) == (max_length,)
assert len(dataset) == len(range(0, len(token_ids) - max_length, stride))

for indice in range(len(dataset)):
    entrada, alvo = dataset[indice]
    assert torch.equal(alvo[:-1], entrada[1:])

# O Dataset rejeita configuracoes que nao podem formar janelas validas.
print("\nVALIDACAO DE PARAMETROS")
for nome, valor_maximo, valor_stride in [
    ("max_length zero", 0, 1),
    ("stride zero", 3, 0),
    ("texto curto", 6, 1),
]:
    try:
        GPTDatasetV1(texto if nome != "texto curto" else "um dois", tokenizador, valor_maximo, valor_stride)
    except ValueError as erro:
        print(f"[OK] {nome}: {erro}")
    else:
        raise AssertionError(f"A validacao deveria falhar: {nome}")

# Uma amostra do corpus real mostra o tamanho que o Dataset pode atingir.
print("\nAMOSTRA DO CORPUS REAL")
caminho = PROJECT_ROOT / "data" / "comptia_security_pluse_701" / "cleaned_data.txt"
texto_corpus = load_text(caminho)
tokenizador_corpus = create_tokenizer(texto_corpus)
dataset_corpus = GPTDatasetV1(texto_corpus, tokenizador_corpus, max_length=8, stride=8)
print("Quantidade de tokens:", len(tokenize(texto_corpus)))
print("Quantidade de amostras:", len(dataset_corpus))
print("Primeira entrada:", dataset_corpus[0][0].tolist())
print("Primeiro alvo:", dataset_corpus[0][1].tolist())
