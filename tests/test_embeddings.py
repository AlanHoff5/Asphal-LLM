# Teste dos embeddings - itens 3.4 e 3.5 da Sprint 2

import sys
from pathlib import Path

import torch
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.embeddings import GPTInputEmbeddings

def test_embeddings():
    # Cada numero representa um Token ID produzido anteriormente pelo tokenizer.
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)

    vocab_size = 10
    context_length = 4
    embedding_dim = 3

    # A semente deixa os valores aleatorios iguais em cada execucao do teste.
    torch.manual_seed(7)
    camada = GPTInputEmbeddings(vocab_size, context_length, embedding_dim)

    print("TESTE DOS EMBEDDINGS")
    print("\nEntrada (Token IDs):")
    print(token_ids)
    print("Shape da entrada:", tuple(token_ids.shape))

    # Item 3.4: cada Token ID consulta um vetor na tabela de tokens.
    token_embeddings = camada.token_embedding(token_ids)

    # Item 3.5: cada posicao consulta um vetor na tabela de posicoes.
    posicoes = torch.arange(context_length)
    positional_embeddings = camada.position_embedding(posicoes)

    # O modelo recebe a soma: conteudo do token mais sua posicao na sequencia.
    input_embeddings = camada(token_ids)

    # item 3.4: cada Token ID consulta um vetor na tabela de tokens.
    print("\nToken Embeddings:")
    print(token_embeddings)
    print("Shape:", tuple(token_embeddings.shape))

    # item 3.5: cada posicao consulta um vetor na tabela de posicoes.
    print("\nPositional Embeddings:")
    print(positional_embeddings)
    print("Shape:", tuple(positional_embeddings.shape))

    print("\nInput Embeddings enviados ao Transformer:")
    print(input_embeddings)
    print("Shape final:", tuple(input_embeddings.shape))

    # O shape segue: (quantidade de sequencias, contexto, dimensao do vetor).
    assert tuple(token_embeddings.shape) == (2, 4, 3)
    assert tuple(positional_embeddings.shape) == (4, 3)
    assert tuple(input_embeddings.shape) == (2, 4, 3)

    # A saida deve ser exatamente a soma dos embeddings de token e posicao.
    assert torch.allclose(
        input_embeddings,
        token_embeddings + positional_embeddings,
    )

def test_posicao_altera_a_representacao():
    # Item 3.5: o MESMO Token ID em posicoes diferentes deve produzir vetores
    # de entrada diferentes, porque o positional embedding e somado ao token
    # embedding. Sem essa soma, a atencao (Sprint 3) nao teria nocao de ordem.
    vocab_size = 10
    context_length = 4
    embedding_dim = 8

    torch.manual_seed(7)
    camada = GPTInputEmbeddings(vocab_size, context_length, embedding_dim)

    # Uma unica sequencia com o token 5 repetido em todas as posicoes.
    token_ids = torch.tensor([[5, 5, 5, 5]], dtype=torch.long)
    saida = camada(token_ids)[0]  # shape (4, embedding_dim)

    print("Token embedding (posicao-independente) do ID 5:")
    print(camada.token_embedding(torch.tensor([5]))[0])
    print("Entrada final por posicao (mesmo token, posicoes 0..3):")
    print(saida)

    # O token embedding puro e identico nas 4 posicoes...
    tok = camada.token_embedding(token_ids)[0]
    assert torch.allclose(tok[0], tok[1]) and torch.allclose(tok[0], tok[3])

    # ...mas a entrada final (token + posicao) difere entre todas as posicoes.
    for i in range(context_length):
        for j in range(i + 1, context_length):
            assert not torch.allclose(saida[i], saida[j])

    # A diferenca entre duas posicoes e exatamente a diferenca dos positional
    # embeddings correspondentes.
    posicoes = camada.position_embedding(torch.arange(context_length))
    assert torch.allclose(saida[1] - saida[0], posicoes[1] - posicoes[0])


if __name__ == "__main__":
    test_embeddings()
    test_posicao_altera_a_representacao()
