import torch
from torch import nn


class GPTInputEmbeddings(nn.Module):
	"""Combina embeddings treinaveis dos tokens e das posicoes."""

	def __init__(self, vocab_size, context_length, embedding_dim):
		super().__init__()
		self.context_length = context_length
		self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
		self.position_embedding = nn.Embedding(context_length, embedding_dim)

	def forward(self, token_ids):
		if token_ids.ndim != 2:
			raise ValueError("token_ids deve ter shape (batch_size, context_length)")
		if token_ids.shape[1] > self.context_length:
			raise ValueError("A sequencia excede o tamanho de contexto")

		# A posicao informa a ordem; a soma une ordem e identidade do token.
		positions = torch.arange(token_ids.shape[1], device=token_ids.device)
		return self.token_embedding(token_ids) + self.position_embedding(positions)
