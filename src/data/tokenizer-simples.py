# Chapter 2
# Transform raw text into a list of tokens, in the Sebastian's Book, this is made with Regular Expressions.

#################
# 3.1 TOKENIZER #
#################

import re

def tokenizar(texto):
    # separate punctuation and whitespace, keep them as tokens
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', texto)
    tokens = [t.strip() for t in tokens if t.strip()]
    return tokens


# Test the tokenizer with a sample text
caminho = "Asphal-LLM/data/the-verdict.txt"
with open(caminho, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Funcion to tokenize the raw text
tokens = tokenizar(raw_text)
print("Total de tokens:", len(tokens))
print(tokens[:30])


##################################
# 3.2 VOCABULARY AND TOKENS ID's #
##################################

tokens_unicos = sorted(set(tokens))
# special tokens for unknown words and end of text 
tokens_unicos.extend(["<|endoftext|>", "<|unk|>"]) # increase the vocabulary with special tokens

vocab = {tok: i for i, tok in enumerate(tokens_unicos)}
print("Tamanho do vocabulário:", len(vocab))


# Simple Tokenizer -> encode / decode
class TokenizadorSimples:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, texto):
        toks = tokenizar(texto)
        toks = [t if t in self.str_to_int else "<|unk|>" for t in toks]
        return [self.str_to_int[t] for t in toks]

    def decode(self, ids):
        texto = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.:;?!"()\'])', r'\1', texto)


# ---------- TESTES ----------
tokenizer = TokenizadorSimples(vocab)

# Test 1: roundtrip (texto -> IDs -> texto)
trecho = raw_text[:200]
ids = tokenizer.encode(trecho)
print("Original:\n", trecho)
print("\nIDs:\n", ids[:30])
print("\nReconstruído:\n", tokenizer.decode(ids))

# Test 2: unknown words, doesnt break anymore - become into <|unk|>
frase_nova = "Isto é um teste com palavraNova"
ids_desc = tokenizer.encode(frase_nova)
print("\nIDs (com palavra nova):", ids_desc)
print("Decodificado:", tokenizer.decode(ids_desc))
# expected: unknown word transformed into <|unk|>

# Test 3: <|endoftext|> 2 separeted texts into 1
texto_a = "Primeiro documento de exemplo."
texto_b = "Segundo documento totalmente separado, fellow."
# Here we use join to concatenate the two texts with the special token <|endoftext|> in between
juntos = " <|endoftext|> ".join([texto_a, texto_b])
ids_join = tokenizer.encode(juntos)
print("\nCom <|endoftext|>:", tokenizer.decode(ids_join))