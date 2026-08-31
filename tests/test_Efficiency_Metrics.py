import unittest
from Efficiency_Metrics import fertility, compression, token_premium

class TestFertility(unittest.TestCase):

    def test_fertility_basic(self):
        text = "hello world"       # 2 words
        tokens = ["hello", "world"]  # 2 tokens
        self.assertEqual(fertility(text, tokens), 1.0)

    def test_fertility_subword_split(self):
        text = "hello"             # 1 word
        tokens = ["hel", "lo"]     # 2 tokens
        self.assertEqual(fertility(text, tokens), 2.0)

    def test_fertility_empty_text(self):
        # word_count = 0 then  should return 0.0, not raise ZeroDivisionError
        self.assertEqual(fertility("", []), 0.0)

    def test_fertility_multiword_multitoken(self):
        text = "the quick brown fox"   # 4 words
        tokens = ["the", "quick", "br", "own", "fox"]  # 5 tokens
        self.assertEqual(fertility(text, tokens), 1.25)


class TestCompression(unittest.TestCase):

    def test_compression_basic(self):
        text = "hi"          # 2 bytes (ASCII)
        tokens = ["hi"]      # 1 token
        self.assertEqual(compression(text, tokens), 2.0)

    def test_compression_multibyte_chars(self):
        # é is 2 bytes in UTF-8, so this tests non-ASCII languages properly
        text = "café"        # c-a-f-é = 1+1+1+2 = 5 bytes
        tokens = ["caf", "é"]  # 2 tokens
        self.assertEqual(compression(text, tokens), 2.5)

    def test_compression_zero_tokens(self):
        # token_count = 0 , then should return 0.0, not raise ZeroDivisionError
        self.assertEqual(compression("hello", []), 0.0)


class TestTokenPremium(unittest.TestCase):

    def test_token_premium_equal(self):
        tokens = ["a", "b", "c"]
        english_tokens = ["a", "b", "c"]
        self.assertEqual(token_premium(tokens, english_tokens), 1.0)

    def test_token_premium_worse(self):
        tokens = ["a", "b", "c", "d"]      # 4 tokens
        english_tokens = ["a", "b"]        # 2 tokens
        self.assertEqual(token_premium(tokens, english_tokens), 2.0)

    def test_token_premium_better(self):
        # target language needs FEWER tokens than english
        tokens = ["a"]
        english_tokens = ["a", "b", "c", "d"]
        self.assertEqual(token_premium(tokens, english_tokens), 0.25)

    def test_token_premium_zero_english_tokens(self):
        # token_English = 0  then should return 0.0, not raise ZeroDivisionError
        self.assertEqual(token_premium(["a", "b"], []), 0.0)


if __name__ == "__main__":
    unittest.main()