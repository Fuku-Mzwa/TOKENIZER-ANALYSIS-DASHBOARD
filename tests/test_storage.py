"""
Unit tests for the Tokenizer Analysis Dashboard (TAD) Storage Engine.
Tests cover:
- Tier 1: Feature coverage (save_leaderboard, load_leaderboard, add_or_update_entry, remove_entry, clear_leaderboard).
- Tier 2: Boundary cases (non-existent file, corrupted JSON recovery, empty list, schema validation, persistence verification).

"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path for direct test execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import storage
except ModuleNotFoundError:
    storage = None


# ===========================================================================
# Tier 1: Feature Coverage — Basic CRUD Operations
# ===========================================================================

@unittest.skipIf(storage is None, "storage module not found (Pending Milestone M3)")
class TestStorageBasicCRUD(unittest.TestCase):
    """
    Tier 1 tests for storage CRUD operations using isolated temporary files.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.temp_dir, "test_leaderboard.json")
        self.sample_entries = [
            {
                "model_id": "bert-base-multilingual-cased",
                "added_at": "2026-09-18T01:00:00Z",
                "languages": ["isiZulu", "isiXhosa"],
                "metrics": {
                    "token_count": 1420,
                    "fertility": 2.15,
                    "compression": 3.85,
                    "token_premium": 1.45,
                    "mbi_precision": 0.42,
                    "mbi_recall": 0.38,
                    "mbi_f1": 0.40,
                },
                "overall_score": 68.5,
            },
            {
                "model_id": "xlm-roberta-base",
                "added_at": "2026-09-18T01:05:00Z",
                "languages": ["isiZulu", "isiXhosa"],
                "metrics": {
                    "token_count": 1250,
                    "fertility": 1.95,
                    "compression": 4.10,
                    "token_premium": 1.25,
                    "mbi_precision": 0.55,
                    "mbi_recall": 0.48,
                    "mbi_f1": 0.51,
                },
                "overall_score": 77.2,
            },
        ]

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_and_load_leaderboard(self):
        """save_leaderboard persists entries to disk; load_leaderboard reads them back."""
        save_ok = storage.save_leaderboard(self.sample_entries, storage_path=self.storage_path)
        self.assertTrue(save_ok)
        self.assertTrue(os.path.exists(self.storage_path))

        loaded = storage.load_leaderboard(storage_path=self.storage_path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["model_id"], "bert-base-multilingual-cased")
        self.assertEqual(loaded[1]["model_id"], "xlm-roberta-base")
        self.assertAlmostEqual(loaded[0]["overall_score"], 68.5, places=2)

    def test_add_new_entry(self):
        """add_or_update_entry adds a new tokenizer entry to the leaderboard."""
        new_entry = {
            "model_id": "gpt2",
            "added_at": "2026-09-18T01:10:00Z",
            "languages": ["English"],
            "metrics": {
                "token_count": 980,
                "fertility": 1.5,
                "compression": 4.5,
                "token_premium": 1.0,
            },
            "overall_score": 82.0,
        }
        res = storage.add_or_update_entry(new_entry, storage_path=self.storage_path)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["model_id"], "gpt2")

        # Verify persisted on disk
        persisted = storage.load_leaderboard(storage_path=self.storage_path)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0]["model_id"], "gpt2")

    def test_update_existing_entry(self):
        """add_or_update_entry updates existing model_id without creating a duplicate."""
        storage.save_leaderboard(self.sample_entries, storage_path=self.storage_path)
        
        # Update bert-base-multilingual-cased with new score
        updated_entry = {
            "model_id": "bert-base-multilingual-cased",
            "added_at": "2026-09-18T01:00:00Z",
            "languages": ["isiZulu", "isiXhosa", "Sesotho"],
            "metrics": {
                "token_count": 1400,
                "fertility": 2.10,
                "compression": 3.90,
                "token_premium": 1.40,
                "mbi_precision": 0.45,
                "mbi_recall": 0.40,
                "mbi_f1": 0.42,
            },
            "overall_score": 72.0,
        }
        res = storage.add_or_update_entry(updated_entry, storage_path=self.storage_path)
        self.assertEqual(len(res), 2, "Duplicate entry created instead of updating existing entry")
        
        # Verify updated in disk storage
        disk_entries = storage.load_leaderboard(storage_path=self.storage_path)
        bert = next(e for e in disk_entries if e["model_id"] == "bert-base-multilingual-cased")
        self.assertEqual(bert["overall_score"], 72.0)
        self.assertEqual(len(bert["languages"]), 3)

    def test_remove_existing_entry(self):
        """remove_entry removes entry matching model_id and persists to disk."""
        storage.save_leaderboard(self.sample_entries, storage_path=self.storage_path)
        res = storage.remove_entry("bert-base-multilingual-cased", storage_path=self.storage_path)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["model_id"], "xlm-roberta-base")

        # Verify on disk
        disk_entries = storage.load_leaderboard(storage_path=self.storage_path)
        self.assertEqual(len(disk_entries), 1)
        self.assertEqual(disk_entries[0]["model_id"], "xlm-roberta-base")

    def test_remove_nonexistent_entry(self):
        """remove_entry with unknown model_id leaves leaderboard intact without error."""
        storage.save_leaderboard(self.sample_entries, storage_path=self.storage_path)
        res = storage.remove_entry("non_existent_model_id", storage_path=self.storage_path)
        self.assertEqual(len(res), 2)

    def test_clear_leaderboard(self):
        """clear_leaderboard removes all entries from disk, returning an empty store."""
        storage.save_leaderboard(self.sample_entries, storage_path=self.storage_path)
        clear_ok = storage.clear_leaderboard(storage_path=self.storage_path)
        self.assertTrue(clear_ok)

        loaded = storage.load_leaderboard(storage_path=self.storage_path)
        self.assertEqual(loaded, [])


# ===========================================================================
# Tier 1: Feature Coverage — Persistence & Schema Validation
# ===========================================================================

@unittest.skipIf(storage is None, "storage module not found (Pending Milestone M3)")
class TestStoragePersistenceAndSchema(unittest.TestCase):
    """
    Tier 1 tests verifying schema preservation and persistence durability.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.temp_dir, "leaderboard.json")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_schema_preserves_nested_structures(self):
        """Verify complex per-language metrics and nested structures survive serialization."""
        entry = {
            "model_id": "meta-llama/Llama-3.2-1B",
            "added_at": "2026-09-18T02:00:00Z",
            "languages": ["isiZulu", "isiXhosa"],
            "summary_metrics": {
                "avg_fertility": 2.10,
                "avg_compression": 3.95,
            },
            "per_language_metrics": {
                "isiZulu": {"fertility": 2.20, "compression": 3.80, "mbi_f1": 0.45},
                "isiXhosa": {"fertility": 2.00, "compression": 4.10, "mbi_f1": 0.49},
            },
            "overall_score": 74.5,
        }
        storage.save_leaderboard([entry], storage_path=self.storage_path)
        loaded = storage.load_leaderboard(storage_path=self.storage_path)
        self.assertEqual(len(loaded), 1)
        loaded_entry = loaded[0]
        self.assertIn("per_language_metrics", loaded_entry)
        self.assertIn("isiZulu", loaded_entry["per_language_metrics"])
        self.assertEqual(loaded_entry["per_language_metrics"]["isiZulu"]["mbi_f1"], 0.45)

    def test_storage_file_is_valid_json_on_disk(self):
        """Storage file on disk must be human-readable and standard JSON."""
        entries = [{"model_id": "test-model", "overall_score": 50.0}]
        storage.save_leaderboard(entries, storage_path=self.storage_path)
        
        with open(self.storage_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.assertIsNotNone(raw_data)


# ===========================================================================
# Tier 2: Boundary Cases & Fault Tolerance
# ===========================================================================

@unittest.skipIf(storage is None, "storage module not found (Pending Milestone M3)")
class TestStorageBoundariesAndFaultTolerance(unittest.TestCase):
    """
    Tier 2 boundary cases:
    - Non-existent file.
    - Corrupted JSON recovery.
    - Empty file (0 bytes).
    - Unexpected root data types.
    - Automatic directory creation.
    - Special characters in model_id.
    - Sequential rapid writes.
    - Large payload persistence.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.temp_dir, "test_file.json")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_load_nonexistent_file(self):
        """Loading a non-existent file must return [] rather than raising FileNotFoundError."""
        non_existent_path = os.path.join(self.temp_dir, "does_not_exist.json")
        res = storage.load_leaderboard(storage_path=non_existent_path)
        self.assertEqual(res, [])

    def test_load_corrupted_json_syntax_error(self):
        """Loading a corrupted JSON file must recover gracefully and return [] without crashing."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json syntax ... truncate")

        res = storage.load_leaderboard(storage_path=self.storage_path)
        self.assertEqual(res, [])

    def test_load_empty_file_zero_bytes(self):
        """Loading a 0-byte file must return [] without raising JSONDecodeError."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            pass  # 0 bytes

        res = storage.load_leaderboard(storage_path=self.storage_path)
        self.assertEqual(res, [])

    def test_load_unexpected_root_type(self):
        """If JSON contains a primitive or unexpected type (e.g. integer or string), return []."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            f.write("42")

        res = storage.load_leaderboard(storage_path=self.storage_path)
        self.assertIsInstance(res, list)

    def test_save_empty_list(self):
        """save_leaderboard([]) must cleanly persist an empty store."""
        ok = storage.save_leaderboard([], storage_path=self.storage_path)
        self.assertTrue(ok)
        loaded = storage.load_leaderboard(storage_path=self.storage_path)
        self.assertEqual(loaded, [])

    def test_auto_create_nested_parent_directories(self):
        """Saving to a deeply nested directory path creates intermediate directories."""
        deep_path = os.path.join(self.temp_dir, "level1", "level2", "sub", "store.json")
        entries = [{"model_id": "nested_model", "overall_score": 60.0}]
        ok = storage.save_leaderboard(entries, storage_path=deep_path)
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(deep_path))

        loaded = storage.load_leaderboard(storage_path=deep_path)
        self.assertEqual(len(loaded), 1)

    def test_model_id_with_special_characters(self):
        """Model IDs with slashes, colons, dots, and hyphens must be stored faithfully."""
        complex_ids = [
            "google-t5/t5-base",
            "meta-llama/Llama-3.2-1B-Instruct",
            "organization.sub/model:v2.0-beta_test",
        ]
        entries = [{"model_id": mid, "overall_score": 70.0} for mid in complex_ids]
        storage.save_leaderboard(entries, storage_path=self.storage_path)
        loaded = storage.load_leaderboard(storage_path=self.storage_path)
        loaded_ids = [e["model_id"] for e in loaded]
        for mid in complex_ids:
            self.assertIn(mid, loaded_ids)

    def test_sequential_updates_integrity(self):
        """Executing multiple add_or_update_entry operations sequentially maintains valid state."""
        for i in range(10):
            entry = {"model_id": f"model_{i}", "overall_score": float(i * 10)}
            storage.add_or_update_entry(entry, storage_path=self.storage_path)

        loaded = storage.load_leaderboard(storage_path=self.storage_path)
        self.assertEqual(len(loaded), 10)

    def test_large_payload_handling(self):
        """Leaderboard with 50 tokenizers serializes and reloads accurately."""
        entries = [
            {
                "model_id": f"model_bench_{i:03d}",
                "metrics": {
                    "fertility": 1.0 + (i * 0.05),
                    "compression": 2.0 + (i * 0.08),
                    "mbi_f1": 0.5 + (i * 0.005),
                },
                "overall_score": 50.0 + (i * 0.5),
            }
            for i in range(50)
        ]
        storage.save_leaderboard(entries, storage_path=self.storage_path)
        loaded = storage.load_leaderboard(storage_path=self.storage_path)
        self.assertEqual(len(loaded), 50)
        self.assertEqual(loaded[49]["model_id"], "model_bench_049")


if __name__ == "__main__":
    from report_runner import run_and_save

    run_and_save(__file__)

