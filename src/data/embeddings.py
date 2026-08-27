import torch
from torch import nn


class GPTInputEmbeddings(nn.Module):
	# Combina a identidade e a posicao de cada token.

	# No Capitulo 2, a Token Embedding transforma cada Token ID em um vetor
	# denso. A Positional Embedding faz o mesmo para as posicoes da sequencia.
	# A soma das duas produz o Input Embedding enviado ao Transformer.

	def __init__(self, vocab_size, context_length, embedding_dim):
		super().__init__()
		# vocab_size e a quantidade de tokens conhecidos pelo vocabulario.
		# context_length e o maior numero de posicoes da sequencia.
		# embedding_dim e o tamanho de cada vetor aprendido.
		self.context_length = context_length

		# Esta tabela consulta um vetor para cada Token ID recebido.
		self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

		# Esta tabela consulta um vetor para cada posicao: 0, 1, 2, etc.
		self.position_embedding = nn.Embedding(context_length, embedding_dim)

	def forward(self, token_ids):
		# A entrada precisa ser um lote de sequencias: (batch_size, context_length).
		if token_ids.ndim != 2:
			raise ValueError("token_ids deve ter shape (batch_size, context_length)")
		if token_ids.shape[1] > self.context_length:
			raise ValueError("A sequencia excede o tamanho de contexto")

		# Gera as posicoes da sequencia no mesmo dispositivo dos Token IDs.
		positions = torch.arange(token_ids.shape[1], device=token_ids.device)

		# token_vectors tem shape (batch_size, context_length, embedding_dim).
		# position_vectors tem shape (context_length, embedding_dim). O PyTorch
		# repete essas posicoes em cada item do lote durante a soma (broadcasting).
		# Assim, cada vetor final representa "qual token" e "em qual posicao".
		token_vectors = self.token_embedding(token_ids)
		position_vectors = self.position_embedding(positions)
		return token_vectors + position_vectors
