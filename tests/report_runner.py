"""
Shared unittest runner that writes a human-readable report to test_reports/.

Usage (at bottom of any test module)::

    if __name__ == "__main__":
        from report_runner import run_and_save
        run_and_save(__file__)

Report format
-------------
TEST FILE: test_Efficiency_Metrics.py
TESTING FRAMEWORK: Python unittest
TESTS EXECUTED: 11
TESTS PASSED: 11
TESTS FAILED: 0
test_... (...) ... ok
----------------------------------------------------------------------
Ran 11 tests in 0.001s
OK
"""
from __future__ import annotations

import sys
import time
import unittest
from io import StringIO
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _reports_dir() -> Path:
    d = _project_root() / "test_reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _report_path_for(test_file: str) -> Path:
    name = Path(test_file).stem
    if name.startswith("test_"):
        stem = name[len("test_") :]
        filename = f"{stem}_test_report.txt"
    else:
        filename = f"{name}_test_report.txt"
    return _reports_dir() / filename


def run_and_save(
    test_file: str,
    *,
    verbosity: int = 2,
    exit: bool = True,
) -> unittest.TestResult:
    """Run tests from the calling module and save a report under test_reports/."""
    test_path = Path(test_file).resolve()
    report_path = _report_path_for(str(test_path))

    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)

    loader = unittest.TestLoader()

    main_mod = sys.modules.get("__main__")
    if (
        main_mod is not None
        and getattr(main_mod, "__file__", None)
        and Path(main_mod.__file__).resolve() == test_path
    ):
        suite = loader.loadTestsFromModule(main_mod)
    else:
        import importlib.util

        module_name = test_path.stem
        spec = importlib.util.spec_from_file_location(module_name, str(test_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load tests from {test_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        suite = loader.loadTestsFromModule(mod)

    buffer = StringIO()
    runner = unittest.TextTestRunner(stream=buffer, verbosity=verbosity)

    start = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - start

    body = buffer.getvalue()
    executed = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = executed - failed - len(result.skipped)

    header_lines = [
        f"TEST FILE: {test_path.name}",
        "TESTING FRAMEWORK: Python unittest",
        f"TESTS EXECUTED: {executed}",
        f"TESTS PASSED: {passed}",
        f"TESTS FAILED: {failed}",
    ]
    if result.skipped:
        header_lines.append(f"TESTS SKIPPED: {len(result.skipped)}")

    cleaned = []
    for line in body.rstrip().splitlines():
        s = line.strip()
        if line.startswith("Ran ") and " test" in line:
            continue
        if s in {"OK", "FAILED"}:
            continue
        if s.startswith("FAILED ("):
            continue
        if s and set(s) <= {"-"}:
            continue
        if not s:
            continue
        cleaned.append(line)

    status = "OK" if result.wasSuccessful() else "FAILED"
    footer = [
        "----------------------------------------------------------------------",
        f"Ran {executed} tests in {elapsed:.3f}s",
        status,
    ]

    report_text = "\n".join(header_lines + cleaned + footer) + "\n"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    sys.stdout.write(report_text)
    sys.stdout.write(f"\n[report saved → {report_path}]\n")

    if exit:
        sys.exit(0 if result.wasSuccessful() else 1)
    return result
