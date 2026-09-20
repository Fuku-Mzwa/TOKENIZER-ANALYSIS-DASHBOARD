"""
Unit tests for the Tokenizer Analysis Dashboard (TAD) Ranking Engine.
Tests cover:
- Tier 1: Feature coverage (normalization, directional inversion, composite weighting, rank ordering).
- Tier 2: Boundary & corner cases (zero values, single tokenizer, missing morphology fallback, extreme scales).

"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path for direct test execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ranking_engine
except ModuleNotFoundError:
    ranking_engine = None


# ===========================================================================
# Tier 1: Feature Coverage — Metric Normalization
# ===========================================================================

@unittest.skipIf(ranking_engine is None, "ranking_engine module not found (Pending Milestone M2)")
class TestMetricNormalization(unittest.TestCase):
    """
    Tier 1 tests for normalize_metric.
    Verifies that raw metrics across disparate numerical scales are scaled
    to [0.0, 1.0] using reference anchor bounds.
    """

    def test_normalize_fertility_midpoint(self):
        """Fertility standard range is typically [1.0, 5.0]. A value of 3.0 should normalize to 0.5."""
        # Using default bounds or custom bounds (1.0, 5.0)
        norm = ranking_engine.normalize_metric(3.0, "fertility", {"fertility": (1.0, 5.0)})
        self.assertAlmostEqual(norm, 0.5, places=3)

    def test_normalize_compression_midpoint(self):
        """Compression range is typically [1.0, 6.0]. A value of 3.5 should normalize to 0.5."""
        norm = ranking_engine.normalize_metric(3.5, "compression", {"compression": (1.0, 6.0)})
        self.assertAlmostEqual(norm, 0.5, places=3)

    def test_normalize_at_minimum_bound(self):
        """A metric value at the lower bound should normalize to exactly 0.0."""
        norm = ranking_engine.normalize_metric(1.0, "fertility", {"fertility": (1.0, 5.0)})
        self.assertAlmostEqual(norm, 0.0, places=3)

    def test_normalize_at_maximum_bound(self):
        """A metric value at the upper bound should normalize to exactly 1.0."""
        norm = ranking_engine.normalize_metric(5.0, "fertility", {"fertility": (1.0, 5.0)})
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_normalize_below_min_clamped(self):
        """Values below the minimum anchor bound must be clamped to 0.0."""
        norm = ranking_engine.normalize_metric(0.5, "fertility", {"fertility": (1.0, 5.0)})
        self.assertEqual(norm, 0.0)

    def test_normalize_above_max_clamped(self):
        """Values above the maximum anchor bound must be clamped to 1.0."""
        norm = ranking_engine.normalize_metric(7.5, "fertility", {"fertility": (1.0, 5.0)})
        self.assertEqual(norm, 1.0)

    def test_normalize_linguistic_metrics(self):
        """MBI metrics (precision, recall, f1) natively live in [0.0, 1.0]."""
        for metric in ["mbi_precision", "mbi_recall", "mbi_f1", "precision", "recall", "f1"]:
            norm = ranking_engine.normalize_metric(0.72, metric)
            self.assertAlmostEqual(norm, 0.72, places=2)

    def test_normalize_token_count_scale(self):
        """Token count typically ranges into thousands. Scale disparity must normalize to [0.0, 1.0]."""
        norm = ranking_engine.normalize_metric(5000.0, "token_count", {"token_count": (0.0, 10000.0)})
        self.assertAlmostEqual(norm, 0.5, places=3)


# ===========================================================================
# Tier 1: Feature Coverage — Directionality Inversion
# ===========================================================================

@unittest.skipIf(ranking_engine is None, "ranking_engine module not found (Pending Milestone M2)")
class TestDirectionalityInversion(unittest.TestCase):
    """
    Tier 1 tests for invert_if_lower_is_better.
    Ensures lower-is-better metrics (Token count, Fertility, Token premium)
    are inverted (1.0 - x) so high scores universally reflect desirability.
    """

    def test_invert_fertility_lower_is_better(self):
        """Fertility: lower is better. Normalized 0.2 (low fertility) -> 0.8 (desirable)."""
        inverted = ranking_engine.invert_if_lower_is_better(0.2, "fertility")
        self.assertAlmostEqual(inverted, 0.8, places=3)

    def test_invert_token_count_lower_is_better(self):
        """Token count: lower is better. Normalized 0.1 -> 0.9."""
        inverted = ranking_engine.invert_if_lower_is_better(0.1, "token_count")
        self.assertAlmostEqual(inverted, 0.9, places=3)

    def test_invert_token_premium_lower_is_better(self):
        """Token premium: lower is better. Normalized 0.35 -> 0.65."""
        inverted = ranking_engine.invert_if_lower_is_better(0.35, "token_premium")
        self.assertAlmostEqual(inverted, 0.65, places=3)

    def test_preserve_compression_higher_is_better(self):
        """Compression: higher is better. Value must NOT be inverted."""
        inverted = ranking_engine.invert_if_lower_is_better(0.85, "compression")
        self.assertAlmostEqual(inverted, 0.85, places=3)

    def test_preserve_mbi_f1_higher_is_better(self):
        """MBI F1: higher is better. Value must NOT be inverted."""
        for metric in ["mbi_f1", "f1"]:
            inverted = ranking_engine.invert_if_lower_is_better(0.68, metric)
            self.assertAlmostEqual(inverted, 0.68, places=3)

    def test_preserve_mbi_precision_and_recall_higher_is_better(self):
        """MBI Precision and Recall: higher is better."""
        for metric in ["mbi_precision", "precision", "mbi_recall", "recall"]:
            inverted = ranking_engine.invert_if_lower_is_better(0.55, metric)
            self.assertAlmostEqual(inverted, 0.55, places=3)

    def test_inversion_boundary_extremes(self):
        """Check 0.0, 0.5, and 1.0 boundary values for lower-is-better."""
        self.assertAlmostEqual(ranking_engine.invert_if_lower_is_better(0.0, "fertility"), 1.0)
        self.assertAlmostEqual(ranking_engine.invert_if_lower_is_better(1.0, "fertility"), 0.0)
        self.assertAlmostEqual(ranking_engine.invert_if_lower_is_better(0.5, "fertility"), 0.5)


# ===========================================================================
# Tier 1: Feature Coverage — Composite Weighting & Transparency
# ===========================================================================

@unittest.skipIf(ranking_engine is None, "ranking_engine module not found (Pending Milestone M2)")
class TestCompositeWeighting(unittest.TestCase):
    """
    Tier 1 tests for compute_composite_score.
    Verifies 50% efficiency / 50% linguistic composite scoring and transparent breakdown.
    """

    def setUp(self):
        self.standard_metrics = {
            "token_count": 1200.0,
            "fertility": 2.0,
            "compression": 4.5,
            "token_premium": 1.2,
            "mbi_precision": 0.60,
            "mbi_recall": 0.50,
            "mbi_f1": 0.55,
        }

    def test_composite_score_breakdown_schema(self):
        """Returns required breakdown dictionary structure per PROJECT.md interface contract."""
        res = ranking_engine.compute_composite_score(self.standard_metrics)
        self.assertIsInstance(res, dict)
        self.assertIn("overall_score", res)
        self.assertIn("efficiency_score", res)
        self.assertIn("linguistic_score", res)
        self.assertIn("normalized_metrics", res)
        self.assertIn("inverted_metrics", res)
        self.assertIn("weights_used", res)
        self.assertIn("missing_components", res)

    def test_composite_score_scale_range(self):
        """Overall score must be within [0.0, 100.0]."""
        res = ranking_engine.compute_composite_score(self.standard_metrics)
        score = res["overall_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_composite_score_equal_weighting_balance(self):
        """When efficiency and linguistic scores are both available, each contributes 50%."""
        res = ranking_engine.compute_composite_score(self.standard_metrics)
        eff = res["efficiency_score"]
        ling = res["linguistic_score"]
        expected_overall = (0.50 * eff) + (0.50 * ling)
        self.assertAlmostEqual(res["overall_score"], expected_overall, places=2)

    def test_superior_tokenizer_scores_higher(self):
        """A tokenizer with lower fertility, higher compression, and higher F1 must score higher."""
        good_metrics = {
            "token_count": 800.0,
            "fertility": 1.4,
            "compression": 5.2,
            "token_premium": 1.05,
            "mbi_precision": 0.85,
            "mbi_recall": 0.80,
            "mbi_f1": 0.82,
        }
        poor_metrics = {
            "token_count": 3500.0,
            "fertility": 4.5,
            "compression": 1.8,
            "token_premium": 3.2,
            "mbi_precision": 0.20,
            "mbi_recall": 0.15,
            "mbi_f1": 0.17,
        }
        score_good = ranking_engine.compute_composite_score(good_metrics)["overall_score"]
        score_poor = ranking_engine.compute_composite_score(poor_metrics)["overall_score"]
        self.assertGreater(score_good, score_poor)

    def test_custom_weights_support(self):
        """Custom weights must be respected when passed."""
        custom_weights = {"efficiency": 0.80, "linguistic": 0.20}
        res = ranking_engine.compute_composite_score(self.standard_metrics, weights=custom_weights)
        eff = res["efficiency_score"]
        ling = res["linguistic_score"]
        expected_overall = (0.80 * eff) + (0.20 * ling)
        self.assertAlmostEqual(res["overall_score"], expected_overall, places=2)


# ===========================================================================
# Tier 1: Feature Coverage — Rank Ordering
# ===========================================================================

@unittest.skipIf(ranking_engine is None, "ranking_engine module not found (Pending Milestone M2)")
class TestRankOrdering(unittest.TestCase):
    """
    Tier 1 tests for rank_tokenizers.
    Verifies descending order sorting and 1-based rank assignment.
    """

    def test_rank_tokenizers_order(self):
        """Entries must be sorted descending by overall_score with ranks 1, 2, 3."""
        entries = [
            {"model_id": "model_b", "metrics": {"fertility": 3.0, "compression": 3.0, "mbi_f1": 0.4}},
            {"model_id": "model_a", "metrics": {"fertility": 1.5, "compression": 5.0, "mbi_f1": 0.8}},
            {"model_id": "model_c", "metrics": {"fertility": 4.5, "compression": 1.5, "mbi_f1": 0.2}},
        ]
        ranked = ranking_engine.rank_tokenizers(entries)
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0]["model_id"], "model_a")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["model_id"], "model_b")
        self.assertEqual(ranked[1]["rank"], 2)
        self.assertEqual(ranked[2]["model_id"], "model_c")
        self.assertEqual(ranked[2]["rank"], 3)

    def test_rank_tokenizers_with_precomputed_scores(self):
        """If overall_score is already present in entries, rank_tokenizers sorts directly."""
        entries = [
            {"model_id": "model_x", "overall_score": 45.0},
            {"model_id": "model_y", "overall_score": 92.5},
            {"model_id": "model_z", "overall_score": 71.0},
        ]
        ranked = ranking_engine.rank_tokenizers(entries)
        self.assertEqual([e["model_id"] for e in ranked], ["model_y", "model_z", "model_x"])
        self.assertEqual([e["rank"] for e in ranked], [1, 2, 3])

    def test_rank_preserves_existing_keys(self):
        """Existing metadata keys (e.g. added_at, languages) must be preserved after ranking."""
        entries = [
            {"model_id": "m1", "overall_score": 80.0, "added_at": "2026-09-18T01:00:00Z", "custom_tag": "test"}
        ]
        ranked = ranking_engine.rank_tokenizers(entries)
        self.assertEqual(ranked[0]["custom_tag"], "test")
        self.assertEqual(ranked[0]["added_at"], "2026-09-18T01:00:00Z")


# ===========================================================================
# Tier 2: Boundary & Corner Cases
# ===========================================================================

@unittest.skipIf(ranking_engine is None, "ranking_engine module not found (Pending Milestone M2)")
class TestRankingBoundariesAndFallbacks(unittest.TestCase):
    """
    Tier 2 boundary cases:
    - Zero values across metrics.
    - Single-tokenizer leaderboard (no div-by-zero).
    - Missing morphology metrics (English/Afrikaans fallback).
    - Degenerate anchor bounds.
    - Extreme scale inputs.
    """

    def test_single_tokenizer_ranking(self):
        """Single-tokenizer leaderboard must assign rank 1 without ZeroDivisionError."""
        entries = [
            {
                "model_id": "lone_model",
                "metrics": {
                    "token_count": 1000.0,
                    "fertility": 2.0,
                    "compression": 3.5,
                    "token_premium": 1.0,
                    "mbi_f1": 0.5,
                },
            }
        ]
        ranked = ranking_engine.rank_tokenizers(entries)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertIn("overall_score", ranked[0])

    def test_empty_leaderboard_ranking(self):
        """Empty list input must return empty list [] without error."""
        ranked = ranking_engine.rank_tokenizers([])
        self.assertEqual(ranked, [])

    def test_missing_linguistic_metrics_fallback(self):
        """
        English and Afrikaans have no NCHLT gold morphology data (mbi_* are None or absent).
        Ranking engine must fall back dynamically to 100% efficiency weighting,
        record 'linguistic' in missing_components, and not crash with TypeError or NaN.
        """
        eff_only_metrics = {
            "token_count": 1200.0,
            "fertility": 2.1,
            "compression": 4.0,
            "token_premium": 1.3,
            # linguistic keys omitted or None
            "mbi_precision": None,
            "mbi_recall": None,
            "mbi_f1": None,
        }
        res = ranking_engine.compute_composite_score(eff_only_metrics)
        self.assertIsNotNone(res["overall_score"])
        self.assertFalse(any(c in [float('nan'), float('inf')] for c in [res["overall_score"]]))
        # Must flag missing linguistic component
        self.assertTrue(
            any("linguistic" in comp.lower() for comp in res["missing_components"]),
            msg=f"missing_components was {res['missing_components']}"
        )
        # Efficiency score should directly inform overall score
        self.assertAlmostEqual(res["overall_score"], res["efficiency_score"], places=2)

    def test_degenerate_normalization_bounds_no_zero_division(self):
        """When min bound equals max bound, normalizer must not raise ZeroDivisionError."""
        norm = ranking_engine.normalize_metric(5.0, "fertility", {"fertility": (5.0, 5.0)})
        self.assertIn(norm, [0.0, 1.0])

    def test_all_zero_metrics_handling(self):
        """Zero values across all metrics should compute cleanly without ZeroDivisionError."""
        zero_metrics = {
            "token_count": 0.0,
            "fertility": 0.0,
            "compression": 0.0,
            "token_premium": 0.0,
            "mbi_precision": 0.0,
            "mbi_recall": 0.0,
            "mbi_f1": 0.0,
        }
        res = ranking_engine.compute_composite_score(zero_metrics)
        self.assertIsInstance(res["overall_score"], (int, float))
        self.assertGreaterEqual(res["overall_score"], 0.0)

    def test_extreme_scale_inputs_no_overflow(self):
        """Massive token counts or compression ratios must clamp cleanly to [0.0, 1.0]."""
        extreme_metrics = {
            "token_count": 10_000_000.0,
            "fertility": 999.0,
            "compression": 500.0,
            "token_premium": 100.0,
            "mbi_precision": 1.0,
            "mbi_recall": 1.0,
            "mbi_f1": 1.0,
        }
        res = ranking_engine.compute_composite_score(extreme_metrics)
        self.assertLessEqual(res["overall_score"], 100.0)
        self.assertGreaterEqual(res["overall_score"], 0.0)

    def test_tied_scores_deterministic_ordering(self):
        """Tokenizers with identical overall_scores must have deterministic ordering."""
        entries = [
            {"model_id": "model_beta", "overall_score": 80.0},
            {"model_id": "model_alpha", "overall_score": 80.0},
        ]
        ranked = ranking_engine.rank_tokenizers(entries)
        self.assertEqual(len(ranked), 2)
        # Ranks must be assigned consistently
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["rank"], 2)


if __name__ == "__main__":
    from report_runner import run_and_save

    run_and_save(__file__)

