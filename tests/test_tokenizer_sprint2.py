# Testes da tokenizacao e do vocabulario - itens 3.1 e 3.2 da Sprint 2

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import (
    SPECIAL_TOKENS,
    build_vocab,
    create_tokenizer,
    load_text,
    tokenize,
)


TEXTO = "Firewall bloqueia ataques."


def test_tokenize_separa_pontuacao_como_token():
    # 3.1: o texto e dividido em unidades menores; a pontuacao vira um token proprio.
    tokens = tokenize(TEXTO)

    print("Texto:", TEXTO)
    print("Tokens:", tokens)
    print("Quantidade de tokens:", len(tokens))

    assert tokens == ["Firewall", "bloqueia", "ataques", "."]
    assert "" not in tokens  # espacos em branco foram removidos


def test_build_vocab_registra_cada_token_uma_vez_com_id_inteiro():
    # 3.2: o vocabulario associa cada token unico a um ID inteiro.
    vocabulario = build_vocab(TEXTO)

    print("Vocabulario:", vocabulario)
    print("Tamanho do vocabulario:", len(vocabulario))

    # 4 tokens do texto + os tokens especiais.
    assert len(vocabulario) == 4 + len(SPECIAL_TOKENS)
    assert all(tok in vocabulario for tok in SPECIAL_TOKENS)
    # Os IDs sao inteiros contiguos comecando em zero.
    assert sorted(vocabulario.values()) == list(range(len(vocabulario)))


def test_encode_decode_reconstroi_o_texto():
    # A relacao Token <-> Token ID <-> Vocabulario permite ida e volta.
    tokenizador = create_tokenizer(TEXTO)
    token_ids = tokenizador.encode(TEXTO)
    texto_recuperado = tokenizador.decode(token_ids)

    print("Token IDs:", token_ids)
    print("Texto recuperado:", texto_recuperado)

    assert all(isinstance(i, int) for i in token_ids)
    assert len(token_ids) == len(tokenize(TEXTO))
    assert texto_recuperado == TEXTO


def test_token_desconhecido_vira_unk_na_versao_2():
    # A V2 substitui por <|unk|> qualquer token ausente no vocabulario.
    tokenizador = create_tokenizer(TEXTO)  # version=2 por padrao
    texto_novo = "Firewall investiga ataques."

    ids_novos = tokenizador.encode(texto_novo)
    decodificado = tokenizador.decode(ids_novos)

    print("Texto novo:", texto_novo)
    print("Token IDs:", ids_novos)
    print("Texto decodificado:", decodificado)

    unk_id = tokenizador.str_to_int["<|unk|>"]
    assert unk_id in ids_novos  # "investiga" nao estava no vocabulario
    assert "<|unk|>" in decodificado


CORPUS_COMPTIA = (
    PROJECT_ROOT / "data" / "comptia_security_pluse_701" / "cleaned_data.txt"
)


@pytest.mark.skipif(
    not CORPUS_COMPTIA.exists(), reason="corpus CompTIA ausente neste checkout"
)
def test_amostra_do_corpus_real_comptia():
    # O mesmo pipeline aplicado ao corpus de Seguranca da Informacao.
    texto_corpus = load_text(CORPUS_COMPTIA)

    tokens = tokenize(texto_corpus)
    tokenizador = create_tokenizer(texto_corpus)

    print("Caracteres:", len(texto_corpus))
    print("Tokens:", len(tokens))
    print("Vocabulario:", len(tokenizador.str_to_int))

    assert len(tokens) > 100_000
    assert len(tokenizador.str_to_int) < len(tokens)  # ha tokens repetidos
    # Todo token do corpus tem um ID e o caminho de volta reconstroi a sequencia.
    ids = tokenizador.encode(texto_corpus)
    assert len(ids) == len(tokens)
    assert [tokenizador.int_to_str[i] for i in ids[:40]] == tokens[:40]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-v"]))
