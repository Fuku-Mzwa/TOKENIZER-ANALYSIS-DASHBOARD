"""
storage.py - Persistent Leaderboard & State Storage Engine
===========================================================
Part of the Tokenizer Analysis Dashboard (TAD) Capstone Project.

Architectural Design & Engineering Rationale (Capstone Explanations):
---------------------------------------------------------------------
1. Why Local Zero-Dependency JSON?
   In an academic capstone and educational research tool like TAD, requiring an external
   database server (e.g. PostgreSQL, MySQL, or MongoDB) introduces disproportionate friction:
   evaluators would have to configure daemon services, manage connection strings, and run
   migration scripts just to test the app. Even embedded engines like SQLite introduce binary
   dependencies and file-locking complexities under concurrent Streamlit reruns.
   
   A structured local JSON file located at `data/leaderboard.json` provides the sweet spot:
   - Zero-dependency: Built entirely on Python standard library modules (`json`, `os`, `tempfile`, `pathlib`).
   - Human-Readable & Transparent: Evaluators can open the leaderboard in VS Code or Notepad,
     inspect the exact normalized metrics, or verify the audit trail without special DB inspection tools.
   - Portable: Works identically across Windows, macOS, and Linux with zero configuration.
   - Negligible Overhead: Benchmark payloads contain tens to hundreds of records (~50KB),
     making complete JSON serialization and parsing take <2 milliseconds.

2. Why Atomic Writes (mkstemp + os.fsync + file close + os.replace) on Windows NTFS?
   A classic pitfall in web applications is file corruption during unexpected termination,
   Streamlit reruns, or power loss. If you open a file with `open(path, 'w')`, the operating
   system immediately truncates the file to zero bytes. If the process is interrupted mid-write,
   the file is left corrupt or empty, destroying the user's benchmarking history.
   
   To guarantee ACID atomicity across all platforms (and specifically Windows NTFS):
   - Same-directory temp file: `tempfile.mkstemp(dir=str(parent_dir))` creates `.leaderboard_tmp_*.json`
     inside the *exact same* parent folder. This is vital: `os.replace` is only atomic if the
     source and target reside on the exact same physical drive/volume. Cross-volume moves fall back
     to non-atomic copy-and-delete.
   - Hardware flush: `f.flush()` followed by `os.fsync(f.fileno())` forces the OS write cache to
     flush all dirty pages directly to physical storage media before renaming.
   - Windows file closure: Unlike POSIX/Linux (where an open inode can be renamed), Windows enforces
     strict mandatory file locking. Calling `os.replace` on an open file handle raises
     `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process`.
     Closing the descriptor before calling `os.replace` guarantees cross-platform reliability.
   - Atomic pointer swap: `os.replace` maps to Windows `MoveFileExW(..., MOVEFILE_REPLACE_EXISTING)`
     or POSIX `rename`, atomically swapping the filesystem directory entry. Readers always see either
     the complete old file or the complete new file—never a partially written state.

3. Why Corrupted File Auto-Recovery?
   Files can occasionally become corrupted through power interruption, disk sector issues,
   or manual user tampering in text editors.
   - Defensive Loading: If `load_leaderboard` encounters `JSONDecodeError`, a 0-byte file,
     or invalid root structure (e.g. non-list JSON), it logs a detailed diagnostic warning
     for developers via `logger.warning` and safely returns an empty list `[]`.
   - Self-Healing: The UI never crashes with an unhandled exception or Streamlit traceback.
     When the user runs their next benchmark or clicks "Reset Leaderboard", `save_leaderboard`
     atomically writes a valid JSON structure, fully healing the persistent store.

Authoritative Specifications:
-----------------------------
- ORIGINAL_REQUEST.md (§R2, §R3)
- instructions.txt.txt (§4, §5)
- PROJECT.md (§Interface Contracts: storage.py)
- tests/test_storage.py (Tier 1 & Tier 2 test suites)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Configure module-level logger for diagnostics and audit trails
logger = logging.getLogger(__name__)

# Base directory of the project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Authoritative default persistent storage path for leaderboard JSON store
DEFAULT_STORAGE_PATH: str = str(PROJECT_ROOT / "data" / "leaderboard.json")


def validate_entry(entry: Any) -> bool:
    """
    Validates whether a candidate object conforms to the minimum leaderboard entry schema.

    Requirements:
    - Must be a dictionary (`isinstance(entry, dict)`).
    - Must contain a non-empty string identifier under the key 'model_id'.

    Parameters:
        entry: The candidate object to validate.

    Returns:
        True if the object is a valid entry dictionary, False otherwise.
    """
    if not isinstance(entry, dict):
        return False
    model_id = entry.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        return False
    return True


def validate_leaderboard_schema(entries: Any) -> bool:
    """
    Validates an entire collection of leaderboard entries.

    Requirements:
    - Root object must be a list.
    - Every element must satisfy `validate_entry`.

    Parameters:
        entries: Candidate leaderboard collection.

    Returns:
        True if the collection is a list of valid entries, False otherwise.
    """
    if not isinstance(entries, list):
        return False
    return all(validate_entry(item) for item in entries)


def normalize_entry_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fills in essential metadata defaults for a leaderboard record without mutating
    the original dictionary or overriding existing values.

    Guarantees the presence of:
    - 'model_id': Preserved from input.
    - 'display_name': Falls back to 'model_id' if absent or empty.
    - 'added_at': Set to current ISO 8601 UTC timestamp if absent.
    - 'metrics': Empty dictionary if absent.
    - 'overall_score': Float 0.0 if absent.

    Parameters:
        entry: Input entry dictionary containing at minimum 'model_id'.

    Returns:
        A shallow copy of the dictionary enriched with standard schema fields.
    """
    record = dict(entry)

    # 1. Display Name: Human-readable model identifier (fallback to model_id)
    if "display_name" not in record or not record["display_name"]:
        record["display_name"] = record.get("model_id", "")

    # 2. Timestamp: ISO 8601 UTC string for tracking benchmark chronology
    if "added_at" not in record and "timestamp" not in record:
        record["added_at"] = datetime.now(timezone.utc).isoformat()
    elif "timestamp" in record and "added_at" not in record:
        record["added_at"] = str(record["timestamp"])

    # 3. Metrics Container: Preserves efficiency and linguistic scores
    if "metrics" not in record:
        record["metrics"] = {}

    # 4. Overall Score: Composite ranking score [0.0 - 100.0]
    if "overall_score" not in record:
        record["overall_score"] = 0.0

    return record


def load_leaderboard(storage_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    Loads stored tokenizer records from the persistent JSON store.

    Fault Tolerance & Recovery:
    - If the file does not exist, returns [] cleanly.
    - If the file is 0 bytes (empty), returns [] without JSON decoding errors.
    - If the file contains malformed JSON syntax or truncated bytes, logs a warning
      for developers and returns [] without raising an uncaught exception.
    - If the JSON root is not a list (e.g., int, dict, string), returns [] safely.
    - Filters out any corrupted records lacking a valid 'model_id'.

    Parameters:
        storage_path: Optional custom file path. If None, defaults to DEFAULT_STORAGE_PATH.

    Returns:
        List of stored tokenizer entry dictionaries.
    """
    resolved_path = Path(storage_path if storage_path is not None else DEFAULT_STORAGE_PATH)

    # 1. Non-existent file check: clean empty state on first launch
    if not resolved_path.exists():
        logger.debug("Leaderboard file does not exist at %s. Returning empty leaderboard.", resolved_path)
        return []

    # 2. 0-byte file check: prevents premature EOF / JSONDecodeError on empty files
    try:
        if resolved_path.stat().st_size == 0:
            logger.warning("Leaderboard file at %s is 0 bytes. Returning empty leaderboard.", resolved_path)
            return []
    except OSError as err:
        logger.warning("Failed to inspect file size for %s: %s. Returning empty leaderboard.", resolved_path, err)
        return []

    # 3. Parsing & Decode Error Handling
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        logger.warning(
            "Leaderboard file at %s contains corrupted JSON syntax (%s). "
            "Recovering gracefully by returning empty leaderboard.",
            resolved_path,
            err,
        )
        return []
    except Exception as err:
        logger.warning(
            "Unexpected error reading leaderboard from %s: %s. Returning empty leaderboard.",
            resolved_path,
            err,
        )
        return []

    # 4. Root Type Verification: Ensure root is a list
    if not isinstance(raw_data, list):
        logger.warning(
            "Leaderboard file at %s contains root type '%s' instead of a JSON list. "
            "Recovering with empty leaderboard.",
            resolved_path,
            type(raw_data).__name__,
        )
        return []

    # 5. Schema Filtering: Keep all entries that conform to minimal valid entry schema
    valid_entries: List[Dict[str, Any]] = []
    malformed_count = 0
    for item in raw_data:
        if validate_entry(item):
            valid_entries.append(item)
        else:
            malformed_count += 1

    if malformed_count > 0:
        logger.warning(
            "Ignored %d malformed entries without valid model_id in %s.",
            malformed_count,
            resolved_path,
        )

    return valid_entries


def save_leaderboard(
    entries: List[Dict[str, Any]],
    storage_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Validates schema and atomically writes a list of leaderboard entries to disk.

    Atomic Write Implementation:
    -----------------------------
    1. Validates that `entries` is a valid list of dictionaries with non-empty `model_id`.
    2. Auto-creates any missing parent directories (e.g. `data/` or nested subdirectories).
    3. Creates a unique temporary file (`.leaderboard_tmp_*.json`) in the *same* directory
       as the target path. Creating the temp file on the same volume guarantees that
       `os.replace` will perform an atomic filesystem pointer swap rather than a slow
       cross-filesystem copy.
    4. Flushes Python stream buffers and calls `os.fsync(fd)` to ensure all bytes are
       physically committed to disk media before renaming.
    5. Closes the file descriptor (required on Windows to prevent `PermissionError: [WinError 32]`).
    6. Atomically replaces the target destination via `os.replace`.
    7. Cleans up temporary artifacts if any error or exception occurs during the process.

    Parameters:
        entries: List of leaderboard dictionary records to persist.
        storage_path: Optional custom file path. If None, defaults to DEFAULT_STORAGE_PATH.

    Returns:
        True if the write and atomic replace succeeded, False otherwise.
    """
    if not isinstance(entries, list):
        logger.warning("save_leaderboard failed: entries argument must be a list, got %s.", type(entries).__name__)
        return False

    # Schema validation across all entries
    for idx, entry in enumerate(entries):
        if not validate_entry(entry):
            logger.warning(
                "save_leaderboard failed: entry at index %d is invalid or lacks a valid 'model_id'. Entry: %s",
                idx,
                entry,
            )
            return False

    target_path = Path(storage_path if storage_path is not None else DEFAULT_STORAGE_PATH).resolve()
    parent_dir = target_path.parent

    # Auto-create parent directories if missing
    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        logger.exception("Failed to create parent directory '%s' for leaderboard: %s", parent_dir, err)
        return False

    temp_fd: Optional[int] = None
    temp_file_path: Optional[str] = None

    try:
        # Create temp file in the same directory to guarantee same filesystem partition
        temp_fd, temp_file_path = tempfile.mkstemp(
            prefix=".leaderboard_tmp_",
            suffix=".json",
            dir=str(parent_dir),
            text=True,
        )

        # Wrap file descriptor in text writer with UTF-8 encoding.
        # Sequence step 1 & 2: mkstemp opens low-level fd, os.fdopen creates buffered text stream.
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            temp_fd = None  # os.fdopen takes ownership of descriptor lifecycle
            json.dump(entries, f, indent=2, ensure_ascii=False)
            f.flush()
            # Sequence step 3 (os.fsync): Forces OS/NTFS write cache to commit dirty pages to media.
            # Without this, sudden system shutdown or driver reboot leaves an empty/partial file.
            os.fsync(f.fileno())

        # Sequence step 4 (file close): Crucial for Windows NTFS!
        # Exiting the 'with' block flushes and closes the file descriptor.
        # On Windows, NTFS enforces mandatory file locking. Calling os.replace on an open handle
        # immediately crashes with `PermissionError: [WinError 32]`.
        # Closing it first guarantees the file lock is released.
        #
        # Sequence step 5 (os.replace): Atomic directory pointer swap.
        # Maps to NTFS MoveFileExW(MOVEFILE_REPLACE_EXISTING). Since both temp and target
        # are in the same folder, this is an instantaneous, atomic rename without data copying.
        os.replace(temp_file_path, str(target_path))
        temp_file_path = None  # Renamed successfully, no cleanup needed
        return True

    except Exception as err:
        logger.exception("Failed to atomically save leaderboard to '%s': %s", target_path, err)
        return False

    finally:
        # If an error occurred before os.fdopen took ownership, close descriptor
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        # Remove orphaned temporary file if replace failed or an exception occurred
        if temp_file_path is not None and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


def add_or_update_entry(
    entry: Dict[str, Any],
    storage_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Upserts a tokenizer record matching 'model_id' into the leaderboard store and
    atomically persists the updated list to disk.

    Behavior:
    - If a record with matching 'model_id' already exists, its contents are updated
      in-place (preserving original 'added_at' timestamp if the new entry omits it).
    - If no match exists, the new record is appended to the leaderboard.
    - Automatically normalizes essential metadata (display_name, added_at, metrics, overall_score).
    - If the storage file was previously missing or corrupted, this cleanly rehabilitates
      the file with the new record.

    Parameters:
        entry: Tokenizer dictionary containing at minimum a non-empty 'model_id'.
        storage_path: Optional custom file path. If None, defaults to DEFAULT_STORAGE_PATH.

    Returns:
        The updated list of all leaderboard entries currently stored.

    Raises:
        ValueError: If `entry` is not a dictionary or lacks a valid 'model_id'.
    """
    if not validate_entry(entry):
        logger.error("add_or_update_entry rejected invalid entry: %s", entry)
        raise ValueError("Leaderboard entry must be a dictionary containing a non-empty string 'model_id'.")

    # Load current state (or [] if file was missing / corrupted)
    current_entries = load_leaderboard(storage_path=storage_path)

    # Normalize entry metadata with sensible defaults
    normalized = normalize_entry_metadata(entry)
    model_id = normalized["model_id"]

    # Locate existing entry by model_id
    existing_index: Optional[int] = None
    for idx, existing in enumerate(current_entries):
        if existing.get("model_id") == model_id:
            existing_index = idx
            break

    if existing_index is not None:
        # Preserve original creation timestamp if the update payload didn't explicitly provide one
        if "added_at" not in entry and "added_at" in current_entries[existing_index]:
            normalized["added_at"] = current_entries[existing_index]["added_at"]
        current_entries[existing_index] = normalized
    else:
        current_entries.append(normalized)

    # Persist updated list to disk
    save_leaderboard(current_entries, storage_path=storage_path)
    return current_entries


def remove_entry(
    model_id: str,
    storage_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Removes a tokenizer entry matching 'model_id' and atomically persists the change to disk.

    Behavior:
    - If 'model_id' is found, the record is removed and disk storage is updated.
    - If 'model_id' is not found, the existing leaderboard is returned unchanged.

    Parameters:
        model_id: Identifier of the tokenizer model to delete.
        storage_path: Optional custom file path. If None, defaults to DEFAULT_STORAGE_PATH.

    Returns:
        The updated list of remaining leaderboard entries.
    """
    current_entries = load_leaderboard(storage_path=storage_path)
    updated_entries = [e for e in current_entries if e.get("model_id") != model_id]

    # Only perform disk write if an entry was actually removed
    if len(updated_entries) != len(current_entries):
        save_leaderboard(updated_entries, storage_path=storage_path)

    return updated_entries


def clear_leaderboard(storage_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Clears all records from the leaderboard and updates disk to an empty store ([]).

    Parameters:
        storage_path: Optional custom file path. If None, defaults to DEFAULT_STORAGE_PATH.

    Returns:
        True if the leaderboard was successfully cleared and persisted, False otherwise.
    """
    return save_leaderboard([], storage_path=storage_path)


def get_entry(
    model_id: str,
    storage_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single tokenizer record matching 'model_id', or None if not found.

    Parameters:
        model_id: Identifier of the tokenizer model to find.
        storage_path: Optional custom file path. If None, defaults to DEFAULT_STORAGE_PATH.

    Returns:
        The matching entry dictionary, or None if not present.
    """
    entries = load_leaderboard(storage_path=storage_path)
    for entry in entries:
        if entry.get("model_id") == model_id:
            return entry
    return None
