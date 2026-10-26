"""Execute every Markdown Python example and compare its documented stdout.

Run from any directory with Python and NumPy installed:
    python scripts/check_note_examples.py

Each Python fence must be immediately followed by a text fence containing its
output. Examples execute in separate namespaces and must import their own inputs.
Unexpected warnings, stderr, missing output blocks, and mismatches fail the run.
Only execute this checker on notes you trust: their Python is executed as code.
"""

import contextlib
import difflib
import io
from pathlib import Path
import re
import sys
import warnings

import numpy as np


FENCE = re.compile(r"^```(\w*)[ \t]*\n(.*?)^```[ \t]*$", re.M | re.S)


def normalize(output):
    """Ignore trailing whitespace, but preserve values and line structure."""
    return "\n".join(line.rstrip() for line in output.rstrip().splitlines())


def check_file(path):
    source = path.read_text(encoding="utf-8")
    blocks = list(FENCE.finditer(source))
    count = 0
    failures = 0

    for index, block in enumerate(blocks):
        if block[1] != "python":
            continue
        count += 1
        line = source.count("\n", 0, block.start()) + 2
        label = f"{path.name}:{line}"
        if (index + 1 == len(blocks) or blocks[index + 1][1] != "text"
                or source[block.end():blocks[index + 1].start()].strip()):
            print(f"FAIL {label}: Python example needs an adjacent text output block")
            failures += 1
            continue

        expected = normalize(blocks[index + 1][2])
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with (contextlib.redirect_stdout(stdout),
                  contextlib.redirect_stderr(stderr),
                  warnings.catch_warnings(),
                  np.printoptions(precision=8, linewidth=75),
                  np.errstate(all="warn")):
                warnings.simplefilter("error")
                # A fresh namespace catches dependencies on earlier examples.
                namespace = {"__name__": "__note_example__"}
                exec(compile(block[2], label, "exec"), namespace)
            actual = normalize(stdout.getvalue())
            if stderr.getvalue():
                raise AssertionError(f"unexpected stderr: {stderr.getvalue()}")
            if actual != expected:
                diff = "\n".join(difflib.unified_diff(
                    expected.splitlines(), actual.splitlines(),
                    fromfile="documented", tofile="executed", lineterm="",
                ))
                raise AssertionError(f"output mismatch\n{diff}")
        except Exception as exc:
            print(f"FAIL {label}: {type(exc).__name__}: {exc}")
            failures += 1

    if not count:
        print(f"FAIL {path.name}: no Python examples found")
        failures += 1
    print(f"{path.name}: {count} examples, {failures} failures")
    return count, failures


def main():
    notes_dir = Path(__file__).resolve().parents[1] / "notes"
    paths = sorted(notes_dir.glob("*.md"))
    if not paths:
        print(f"FAIL: no notes found in {notes_dir}")
        return 1

    count = failures = 0
    for path in paths:
        checked, failed = check_file(path)
        count += checked
        failures += failed
    print(f"NumPy {np.__version__}: {count} examples, {failures} failures")
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
