import re
import tiktoken

# 1) PRIMEIRO define a função
def tokenizar(texto):
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', texto)
    return [t.strip() for t in tokens if t.strip()]

# 2) DEPOIS lê o arquivo e cria raw_text
caminho = "data/the-verdict.txt"          # confira este caminho (ver nota abaixo)
with open(caminho, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 3) SÓ ENTÃO usa raw_text
tokens = tokenizar(raw_text)
print("Total de tokens:", len(tokens))

# ... vocabulário, TokenizadorSimples ...

# 4) BPE — usa raw_text, então tem que vir DEPOIS do passo 2
bpe = tiktoken.get_encoding("gpt2")
ids = bpe.encode(raw_text, allowed_special={"<|endoftext|>"})
print("Total de tokens (BPE):", len(ids))