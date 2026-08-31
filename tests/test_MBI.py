import unittest
from MBI import (
    get_gold_boundaries,
    get_tokenizer_boundaries,
    get_boundary_counts,
    precision,
    recall,
    f1,
)

class TestGoldBoundaries(unittest.TestCase):

    def test_gold_boundaries_basic(self):
        result = get_gold_boundaries("Ad_il_e")
        self.assertEqual(result, {2, 4})

    def test_gold_boundaries_no_split(self):
        result = get_gold_boundaries("Word")
        self.assertEqual(result, set())

    def test_gold_boundaries_single_split(self):
        result = get_gold_boundaries("un_happy")
        self.assertEqual(result, {2})


class TestTokenizerBoundaries(unittest.TestCase):

    def test_tokenizer_boundaries_basic(self):
        result = get_tokenizer_boundaries(["Ad", "il", "e"])
        self.assertEqual(result, {2, 4})

    def test_tokenizer_boundaries_single_token(self):
        # only one token -> tokens[:-1] is empty -> no boundaries at all
        result = get_tokenizer_boundaries(["Adile"])
        self.assertEqual(result, set())

    def test_tokenizer_boundaries_prefix_marker_bug(self):
        # documents the known ▁/Ġ limitation: Ġ is counted as a real character,
        # shifting every boundary position by +1 relative to the true text
        result = get_tokenizer_boundaries(["ĠAd", "il", "e"])
        self.assertEqual(result, {3, 5})  # NOT {2, 4} - this IS the bug


class TestBoundaryCounts(unittest.TestCase):

    def test_perfect_match(self):
        text = "Adile"
        tokens = ["Ad", "il", "e"]
        gold = "Ad_il_e"
        tp, fp, tn, fn = get_boundary_counts(text, tokens, gold)
        self.assertEqual((tp, fp, tn, fn), (2, 0, 2, 0))

    def test_partial_match(self):
        text = "Adile"
        tokens = ["A", "dil", "e"]  # boundaries at {1, 4} vs gold {2, 4}
        gold = "Ad_il_e"
        tp, fp, tn, fn = get_boundary_counts(text, tokens, gold)
        self.assertEqual((tp, fp, tn, fn), (1, 1, 1, 1))

    def test_no_overlap(self):
        text = "abcd"       # possible positions {1,2,3}
        tokens = ["a", "b", "cd"]  # boundaries {1, 2}
        gold = "a_b_c_d"    # gold boundaries {1, 2, 3}
        tp, fp, tn, fn = get_boundary_counts(text, tokens, gold)
        # tp: {1,2} = 2, fp: {} = 0, fn: {3} = 1, tn: {} = 0
        self.assertEqual((tp, fp, tn, fn), (2, 0, 0, 1))


class TestPrecision(unittest.TestCase):

    def test_precision_normal(self):
        self.assertEqual(precision(2, 0), 1.0)

    def test_precision_partial(self):
        self.assertEqual(precision(1, 1), 0.5)

    def test_precision_zero_division_guard(self):
        self.assertEqual(precision(0, 0), 0.0)


class TestRecall(unittest.TestCase):

    def test_recall_normal(self):
        self.assertEqual(recall(2, 0), 1.0)

    def test_recall_partial(self):
        self.assertEqual(recall(1, 1), 0.5)

    def test_recall_zero_division_guard(self):
        self.assertEqual(recall(0, 0), 0.0)


class TestF1(unittest.TestCase):

    def test_f1_perfect(self):
        self.assertEqual(f1(1.0, 1.0), 1.0)

    def test_f1_partial(self):
        self.assertEqual(f1(0.5, 0.5), 0.5)

    def test_f1_zero_division_guard(self):
        self.assertEqual(f1(0.0, 0.0), 0.0)

    def test_f1_end_to_end_partial_match(self):
        # chains precision/recall from the partial match case above
        p = precision(1, 1)
        r = recall(1, 1)
        self.assertEqual(f1(p, r), 0.5)


if __name__ == "__main__":
    unittest.main()