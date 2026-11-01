"""
Runs the whole test suite and shows the results.

Usage (from the project folder):
    python run_tests.py                 run everything
    python run_tests.py unit            only the unit tests
    python run_tests.py integration     only the integration tests

It prints every test as it runs, then a summary table, and saves a report with the
test cases (purpose, test data, expected result) and their results to
tests/test_report.md.

Needs pytest:   pip install pytest
"""
import inspect
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import pytest
except ImportError:
    print("pytest is not installed. Install it with:  pip install pytest")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "tests" / "test_report.md"
DOC_KEYS = ["Purpose", "Test data", "Expected"]


MODEL_SCRIPTS = ["sms_model.py", "email_model.py", "images_model.py", "url_model.py"]


def check_model_scripts():
    """Prints which model script file will be tested for each model, and whether it has the upgrades."""
    sys.path.insert(0, str(ROOT / "tests"))
    from helpers import candidate_scripts  # same lookup the tests use

    print("Checking model scripts...")
    problems = []
    for name in MODEL_SCRIPTS:
        found = candidate_scripts(name)
        if not found:
            print(f"  {name.replace('.py', ''):<14} NOT FOUND (looked in models/ and the project folder)")
            problems.append(name)
            continue
        path = found[0]
        text = path.read_text(encoding="utf-8", errors="replace")
        upgraded = "def predict_with_confidence" in text and "TRAINING STATISTICS" in text
        status = "upgraded" if upgraded else "ORIGINAL VERSION (not upgraded)"
        print(f"  {name.replace('.py', ''):<14} {status:<33} {path.relative_to(ROOT)}")
        if len(found) > 1:
            print(f"  {'':<14} (older copies in the same folder are ignored: "
                  + ", ".join(p.name for p in found[1:]) + ")")
        if not upgraded:
            problems.append(name)

    if problems:
        print("\n  WARNING: " + ", ".join(problems))
        print("  These scripts are missing or are still the original versions, so the tests that")
        print("  use them will fail. Replace them with the upgraded versions and run again.")
    print()
    return problems


class ResultCollector:
    """pytest plugin that remembers every test's description and result."""

    def __init__(self):
        self.docs = {}
        self.types = {}
        self.results = []

    def pytest_collection_modifyitems(self, items):
        for item in items:
            function = getattr(item, "function", None)
            self.docs[item.nodeid] = (inspect.getdoc(function) or "") if function else ""
            self.types[item.nodeid] = "unit" if item.get_closest_marker("unit") else "integration"
        # show unit tests first, then integration tests (keeps the order inside each group)
        items.sort(key=lambda item: 0 if item.get_closest_marker("unit") else 1)

    def pytest_runtest_logreport(self, report):
        # the result of a test is in the "call" phase; a skip/error can also happen in "setup"
        if report.when == "call" or (report.when == "setup" and (report.skipped or report.failed)):
            note = ""
            if report.passed:
                outcome = "PASSED"
            elif report.skipped:
                outcome = "SKIPPED"
                longrepr = report.longrepr
                note = longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)
                note = note.replace("Skipped: ", "")
            else:
                outcome = "FAILED"
                crash = getattr(report.longrepr, "reprcrash", None)
                note = crash.message if crash else str(report.longrepr)[:300]

            self.results.append({
                "id": report.nodeid,
                "outcome": outcome,
                "duration": report.duration,
                "note": note.replace("\n", " ")[:300],
            })


def parse_docstring(doc):
    """Splits a test's docstring into Purpose / Test data / Expected."""
    parsed = {key: "" for key in DOC_KEYS}
    current = None
    for line in doc.splitlines():
        text = line.strip()
        for key in DOC_KEYS:
            if text.startswith(key + ":"):
                current = key
                parsed[key] = text[len(key) + 1:].strip()
                break
        else:
            if current and text:
                parsed[current] += " " + text
    return parsed


def short_name(node_id):
    return node_id.split("tests/", 1)[-1]


def print_summary(results, types):
    print("\n" + "=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)
    print(f"{'Type':<12}{'Result':<10}{'Time':>8}   Test")
    print("-" * 100)
    for item in results:
        print(
            f"{types[item['id']]:<12}{item['outcome']:<10}"
            f"{item['duration']:>7.2f}s   {short_name(item['id'])}"
        )
        if item["outcome"] != "PASSED" and item["note"]:
            print(f"{'':<31}-> {item['note']}")

    counts = {name: sum(r["outcome"] == name for r in results)
              for name in ("PASSED", "FAILED", "SKIPPED")}
    print("-" * 100)
    print(
        f"Total: {len(results)}   Passed: {counts['PASSED']}   "
        f"Failed: {counts['FAILED']}   Skipped: {counts['SKIPPED']}"
    )
    return counts


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def write_report(collector, counts, problems):
    lines = [
        "# Test report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Total:** {len(collector.results)} | **Passed:** {counts['PASSED']} | "
        f"**Failed:** {counts['FAILED']} | **Skipped:** {counts['SKIPPED']}",
        "",
    ]
    if problems:
        lines += ["> **Warning:** these model scripts are missing or not upgraded: "
                  + ", ".join(problems), ""]
    for kind, title in (("unit", "Unit tests"), ("integration", "Integration tests")):
        rows = [r for r in collector.results if collector.types[r["id"]] == kind]
        lines += [f"## {title} ({len(rows)})", "",
                  "| Test case | Purpose | Test data | Expected result | Result |",
                  "|---|---|---|---|---|"]
        for item in rows:
            doc = parse_docstring(collector.docs.get(item["id"], ""))
            result = item["outcome"]
            if item["note"]:
                result += f" - {item['note']}"
            lines.append(
                f"| `{short_name(item['id'])}` | {cell(doc['Purpose'])} | "
                f"{cell(doc['Test data'])} | {cell(doc['Expected'])} | {cell(result)} |"
            )
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    os.chdir(ROOT)

    choice = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if choice not in ("all", "unit", "integration"):
        print("Usage: python run_tests.py [unit|integration]")
        return 2

    tests_dir = ROOT / "tests"
    if not tests_dir.exists():
        print(f"Could not find the tests folder: {tests_dir}")
        return 2

    # the tests are searched for in the whole tests/ folder, whichever sub folder they are in;
    # unit / integration is chosen by the marker each test file sets for itself
    extra = [] if choice == "all" else ["-m", choice]

    problems = check_model_scripts()
    collector = ResultCollector()
    exit_code = pytest.main(["-v", "--tb=short", "-p", "no:cacheprovider", str(tests_dir)] + extra,
                            plugins=[collector])

    if not collector.results:
        print("\nNo tests were found" + ("" if choice == "all" else f" for '{choice}'") + ".")
        print(f"Make sure the test files (test_*.py) are somewhere inside: {tests_dir}")
        print("Expected: tests/unit/test_prediction_format.py and tests/integration/test_*.py")
        return 1

    counts = print_summary(collector.results, collector.types)
    write_report(collector, counts, problems)
    print(f"\nFull report saved to: {REPORT_PATH}")
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
