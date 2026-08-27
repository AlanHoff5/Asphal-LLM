import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
	"""Cria pares de entrada e alvo usando uma janela deslizante."""

	def __init__(self, text, tokenizer, max_length, stride):
		token_ids = tokenizer.encode(text)
		if len(token_ids) <= max_length:
			raise ValueError("O texto precisa ter mais tokens que max_length")
		if max_length <= 0 or stride <= 0:
			raise ValueError("max_length e stride devem ser positivos")

		self.input_ids = []
		self.target_ids = []

		for start in range(0, len(token_ids) - max_length, stride):
			# Cada alvo comeca no token seguinte para treinar a previsao do proximo token.
			self.input_ids.append(torch.tensor(token_ids[start:start + max_length]))
			self.target_ids.append(torch.tensor(token_ids[start + 1:start + max_length + 1]))

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, index):
		return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
	text,
	tokenizer,
	batch_size=4,
	max_length=256,
	stride=128,
	shuffle=True,
	drop_last=True,
):
	"""Cria um DataLoader com lotes de entradas e alvos deslocados."""
	dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
