"""Testes automatizados dos componentes da Sprint 2."""

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GPTDatasetV1, create_dataloader_v1  # noqa: E402
from src.data.embeddings import GPTInputEmbeddings  # noqa: E402
from src.data.tokenizer import (  # noqa: E402
    SPECIAL_TOKENS,
    build_vocab,
    create_tokenizer,
    load_text,
    tokenize,
)


class TestTokenizer(unittest.TestCase):
    def test_tokenize_separates_punctuation(self):
        self.assertEqual(tokenize("Ola, mundo!"), ["Ola", ",", "mundo", "!"])

    def test_vocab_contains_special_tokens(self):
        vocab = build_vocab("um dois")
        self.assertTrue(all(token in vocab for token in SPECIAL_TOKENS))

    def test_tokenizer_roundtrip_and_unknown_token(self):
        tokenizer = create_tokenizer("um dois")
        ids = tokenizer.encode("um palavraNova")
        self.assertEqual(tokenizer.decode(ids), "um <|unk|>")


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.text = "um dois tres quatro cinco seis"
        self.tokenizer = create_tokenizer(self.text)

    def test_target_is_shifted_by_one(self):
        dataset = GPTDatasetV1(self.text, self.tokenizer, max_length=3, stride=1)
        inputs, targets = dataset[0]
        self.assertEqual(targets[:-1].tolist(), inputs[1:].tolist())

    def test_dataloader_returns_expected_shapes(self):
        dataloader = create_dataloader_v1(
            self.text,
            self.tokenizer,
            batch_size=2,
            max_length=3,
            stride=1,
            shuffle=False,
            drop_last=False,
        )
        inputs, targets = next(iter(dataloader))
        self.assertEqual(tuple(inputs.shape), (2, 3))
        self.assertEqual(tuple(targets.shape), (2, 3))


class TestEmbeddings(unittest.TestCase):
    def test_input_embedding_shape(self):
        layer = GPTInputEmbeddings(vocab_size=20, context_length=4, embedding_dim=8)
        output = layer(torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]))
        self.assertEqual(tuple(output.shape), (2, 4, 8))

    def test_sequence_cannot_exceed_context_length(self):
        layer = GPTInputEmbeddings(vocab_size=20, context_length=4, embedding_dim=8)
        with self.assertRaises(ValueError):
            layer(torch.tensor([[1, 2, 3, 4, 5]]))


class TestCorpus(unittest.TestCase):
    def test_compTIA_corpus_can_be_loaded(self):
        path = PROJECT_ROOT / "data" / "comptia_security_pluse_701" / "cleaned_data.txt"
        text = load_text(path)
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
