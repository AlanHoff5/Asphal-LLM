# Chapter 2 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - tokenizer for the cybersecurity corpus

# 3.1 & 3.2

from pathlib import Path
import re


SPECIAL_TOKENS = ("<|endoftext|>", "<|unk|>")
# Tokens especiais para indicar o fim do texto e palavras desconhecidas.


def tokenize(text):
    # Funcao tokenize que recebe um texto como entrada e retorna uma lista de tokens.
    # A funcao utiliza regex para dividir o texto em tokens, considerando pontuacoes
    # e espacos em branco. Em seguida, remove os tokens vazios.
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [token.strip() for token in tokens if token.strip()]


# LEITURA DE ARQUIVO
def load_text(path, encoding="utf-8"):
    # Abre um arquivo de texto e retorna todo o seu conteudo como uma string.
    return Path(path).read_text(encoding=encoding)


# CRIANÇÃO DE VOCABULARIO
def build_vocab(text):
    # Cria o vocabulario a partir de todos os tokens encontrados no corpus.
    # Cada token recebe um numero inteiro, usado posteriormente pelo modelo.
    unique_tokens = sorted(set(tokenize(text)))
    # Tokens especiais para palavras desconhecidas e separacao de textos.
    for special_token in SPECIAL_TOKENS:
        if special_token not in unique_tokens:
            unique_tokens.append(special_token)
    return {token: index for index, token in enumerate(unique_tokens)}


class SimpleTokenizerV1:
    # Tokenizador simples que converte tokens em IDs e IDs de volta para tokens.
    def __init__(self, vocab):
        self.str_to_int = vocab  # Mapeia tokens para inteiros.
        self.int_to_str = {integer: token for token, integer in vocab.items()}
        # Cria o dicionario inverso, mapeando inteiros para tokens.

    def encode(self, text):
        # Converte o texto em tokens e depois em IDs usando o vocabulario.
        tokens = tokenize(text)
        return [self.str_to_int[token] for token in tokens]

    def decode(self, token_ids):
        # Converte os IDs de volta para texto e remove espacos antes da pontuacao.
        text = " ".join(self.int_to_str[token_id] for token_id in token_ids)
        return re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)


class SimpleTokenizerV2(SimpleTokenizerV1):
    # Versao que substitui tokens ausentes no vocabulario por <|unk|>.
    def encode(self, text):
        tokens = tokenize(text)
        tokens = [token if token in self.str_to_int else "<|unk|>" for token in tokens]
        return [self.str_to_int[token] for token in tokens]


def create_tokenizer(text, version=2):
    """Cria um tokenizador cujo vocabulario e baseado no texto fornecido.
    Primeiro, "build_vocab" divide o texto em tokens e atribui um ID inteiro. 
    Depois, esse vocabulario eh entregue a classe do
    tokenizador escolhida. A V2 e usada por padrao porque transforma tokens
    desconhecidos em "<|unk|>" durante a codificacao.
    """
    tokenizer_class = SimpleTokenizerV2 if version == 2 else SimpleTokenizerV1
    # O vocabulario e criado uma vez e passado ao construtor do tokenizador.
    return tokenizer_class(build_vocab(text))


def create_tokenizer_from_file(path, version=2):
    """Le um corpus de um arquivo e cria um tokenizador para ele.
    Esta funcao e apenas uma combinacao de "load_text" e
    "create_tokenizer": le todo o arquivo como uma string e usa essa string
    para montar o vocabulario. O tokenizador retornado conhece os tokens que
    aparecem nesse arquivo; palavras de outro texto podem ser convertidas
    para "<|unk|>" quando a versao 2 for utilizada.
    """
    return create_tokenizer(load_text(path), version=version)


def get_bpe_tokenizer():
    # Retorna um tokenizador BPE GPT-2 pronto, fornecido pela biblioteca tiktoken.
    import tiktoken
    return tiktoken.get_encoding("gpt2")
