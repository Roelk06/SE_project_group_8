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


class ResultCollector:
    """pytest plugin that remembers every test's description and result."""

    def __init__(self):
        self.docs = {}
        self.results = []

    def pytest_collection_modifyitems(self, items):
        for item in items:
            function = getattr(item, "function", None)
            self.docs[item.nodeid] = (inspect.getdoc(function) or "") if function else ""

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


def test_type(node_id):
    return "unit" if "/unit/" in node_id else "integration"


def short_name(node_id):
    return node_id.split("tests/", 1)[-1]


def print_summary(results):
    print("\n" + "=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)
    print(f"{'Type':<12}{'Result':<10}{'Time':>8}   Test")
    print("-" * 100)
    for item in results:
        print(
            f"{test_type(item['id']):<12}{item['outcome']:<10}"
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


def write_report(collector, counts):
    lines = [
        "# Test report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Total:** {len(collector.results)} | **Passed:** {counts['PASSED']} | "
        f"**Failed:** {counts['FAILED']} | **Skipped:** {counts['SKIPPED']}",
        "",
    ]
    for kind, title in (("unit", "Unit tests"), ("integration", "Integration tests")):
        rows = [r for r in collector.results if test_type(r["id"]) == kind]
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
    if choice == "unit":
        targets = [str(ROOT / "tests" / "unit")]
    elif choice == "integration":
        targets = [str(ROOT / "tests" / "integration")]
    elif choice == "all":
        targets = [str(ROOT / "tests" / "unit"), str(ROOT / "tests" / "integration")]
    else:
        print("Usage: python run_tests.py [unit|integration]")
        return 2

    collector = ResultCollector()
    exit_code = pytest.main(["-v", "--tb=short", "-p", "no:cacheprovider"] + targets,
                            plugins=[collector])

    counts = print_summary(collector.results)
    write_report(collector, counts)
    print(f"\nFull report saved to: {REPORT_PATH}")
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
