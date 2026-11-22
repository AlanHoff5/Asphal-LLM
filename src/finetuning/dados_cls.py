# Chapter 6 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM

# 6.2 e 6.3

# Prepara o dataset de spam/ham do livro para o fine-tuning de classificação.

# O dataset original (SMS Spam Collection, UCI) tem duas colunas:
#   - label: "spam" ou "ham" (não-spam)
#   - text:  mensagem de texto

# Pipeline:
#   1. Baixar / carregar o CSV
#   2. Balancear as classes (undersample da maioria)
#   3. Dividir em treino (70%), validação (10%) e teste (20%)
#   4. Tokenizar e criar SpamDataset (trunca ou padding até max_length)
#   5. Criar DataLoaders com collate_fn customizada

# URL do dataset:
#   https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip


import urllib.request
import zipfile
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Download e carregamento |
# -------------------------

URL_SPAM = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00228/smsspamcollection.zip"
)


def baixar_dataset_spam(diretorio: Path) -> Path:
    # Baixa e descompacta o SMS Spam Collection.

    # Retorna o caminho do arquivo TSV (tab-separated).
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)
    arquivo_tsv = diretorio / "SMSSpamCollection"

    if arquivo_tsv.exists():
        print(f"Dataset já existe: {arquivo_tsv}")
        return arquivo_tsv

    zip_path = diretorio / "smsspam.zip"
    print("Baixando SMS Spam Collection...")
    urllib.request.urlretrieve(URL_SPAM, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("SMSSpamCollection", diretorio)
    zip_path.unlink()
    print(f"Dataset salvo em: {arquivo_tsv}")
    return arquivo_tsv


def carregar_spam_df(caminho: Path):
    # Carrega o TSV e devolve uma lista de (rotulo_int, texto).

    # Mapeamento: ham=0, spam=1.
    dados = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            partes = linha.split("\t", 1)
            if len(partes) != 2:
                continue
            label_str, texto = partes
            rotulo = 1 if label_str.strip().lower() == "spam" else 0
            dados.append((rotulo, texto))
    return dados


def balancear_classes(dados):
    # Equaliza o número de exemplos por classe (undersample da maioria).

    # Retorna uma lista balanceada embaralhada de forma determinística.
    spam = [d for d in dados if d[0] == 1]
    ham  = [d for d in dados if d[0] == 0]
    n    = min(len(spam), len(ham))
    # Pega os primeiros n de cada (order natural do arquivo — sem shuffle aqui;
    # o DataLoader de treino faz o shuffle depois)
    return spam[:n] + ham[:n]


def dividir_treino_val_teste(dados, train_frac=0.70, val_frac=0.10):
    # Divide em treino / validação / teste.

    # Retorna (treino, val, teste) como listas de (rotulo, texto).
    n = len(dados)
    i_val  = int(n * train_frac)
    i_test = int(n * (train_frac + val_frac))
    return dados[:i_val], dados[i_val:i_test], dados[i_test:]


# -----------------
# Dataset PyTorch |
# -----------------

class SpamDataset(Dataset):
    # Dataset de spam/ham tokenizado com padding ou truncagem.

    # Cada item é (input_ids, rotulo):
    #   - input_ids: tensor de tamanho fixo `max_length` (pad_id=50256 do GPT-2)
    #   - rotulo: 0 (ham) ou 1 (spam)

    PAD_ID = 50256   # <|endoftext|> do GPT-2 — usado também como padding

    def __init__(self, dados, tokenizer, max_length=None, padding=True):

        # Parâmetros
        # ----------
        # dados : list de (rotulo_int, texto)
        # tokenizer : tiktoken.Encoding
        # max_length : int | None
        #     Se None, usa o comprimento máximo do conjunto.
        # padding : bool
        #     Se True, padeia até max_length. Se False, só trunca.
        self.tokenizer  = tokenizer
        self.rotulos    = []
        self.input_ids  = []

        ids_brutos = [tokenizer.encode(texto) for _, texto in dados]
        rotulos    = [r for r, _ in dados]

        if max_length is None:
            max_length = max(len(ids) for ids in ids_brutos)
        self.max_length = max_length

        for ids, r in zip(ids_brutos, rotulos):
            # Trunca
            ids = ids[:max_length]
            # Padeia
            if padding:
                ids = ids + [self.PAD_ID] * (max_length - len(ids))
            self.input_ids.append(torch.tensor(ids, dtype=torch.long))
            self.rotulos.append(torch.tensor(r, dtype=torch.long))

    def __len__(self):
        return len(self.rotulos)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.rotulos[idx]


# ---------------------
# DataLoaders prontos |
# ---------------------

def criar_loaders_spam(dados_treino, dados_val, dados_teste,
                       tokenizer, max_length, batch_size=8, num_workers=0):
    # Cria os três DataLoaders (treino, validação, teste).

    # max_length deve ser calculado ANTES de chamar esta função:
    #     max_length = min(context_length, max(len(tokenizer.encode(t)) for _, t in dados))

    # O treino usa shuffle=True; val e teste usam shuffle=False.
    ds_treino = SpamDataset(dados_treino, tokenizer, max_length=max_length)
    ds_val    = SpamDataset(dados_val,    tokenizer, max_length=max_length)
    ds_teste  = SpamDataset(dados_teste,  tokenizer, max_length=max_length)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    return (
        DataLoader(ds_treino, shuffle=True,  **kwargs),
        DataLoader(ds_val,    shuffle=False, **kwargs),
        DataLoader(ds_teste,  shuffle=False, **kwargs),
    )
