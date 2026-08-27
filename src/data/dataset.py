# Chapter 2 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - tokenizer for the cybersecurity corpus

# 3.3

import torch
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    # O Dataset organiza amostras para o treinamento e permite que o PyTorch
    # acesse cada par de entrada e alvo por meio de __getitem__.

    def __init__(self, texto, tokenizer, max_length, stride):
        
        # O tokenizer transforma o texto em uma sequencia de Token IDs.
        token_ids = tokenizer.encode(texto) # Tokens IDs

        if max_length <= 0 or stride <= 0:
            raise ValueError("max_length e stride devem ser positivos")
        if len(token_ids) <= max_length:
            raise ValueError("O texto precisa ter mais tokens que max_length")

        # Listas separadas guardam as janelas que serao entregues ao modelo.
        self.input_ids = [] # Entradas: cada janela de max_length tokens do texto
        self.target_ids = [] # Alvos: a mesma janela deslocada para prever o proximo token

        # A janela percorre o texto usando stride. Cada entrada tem max_length
        # tokens; o alvo e a mesma janela deslocada para prever o proximo token.
        for inicio in range(0, len(token_ids) - max_length, stride):
            entrada = token_ids[inicio:inicio + max_length]
            alvo = token_ids[inicio + 1:inicio + max_length + 1]
            self.input_ids.append(torch.tensor(entrada, dtype=torch.long))
            self.target_ids.append(torch.tensor(alvo, dtype=torch.long))

    def __len__(self):
        # Informa ao DataLoader quantas janelas de treinamento foram criadas.
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Retorna uma amostra pronta: entrada e respectivos tokens esperados.
        return self.input_ids[idx], self.target_ids[idx]