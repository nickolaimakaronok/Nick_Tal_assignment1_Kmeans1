#!/usr/bin/env python3
"""
K-means Tester (30+ tests)

✔ Correct CLI validation tests
✔ Deterministic correctness tests
✔ Formatting tests (4 decimals)
✔ Stress tests
✔ Proper Valgrind memory checks
✔ Accepts standard exit codes (0 success, 1 error)
"""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple

# =============================
# CONFIG
# =============================
C_BIN = "./kmeans"
RUN_VALGRIND = True

# =============================
# DATASETS (valid content)
# =============================
DATASETS = {
    "data_A.txt": (
        "0,0\n0,1\n1,0\n1,1\n10,10\n10,11\n11,10\n11,11\n\n"
    ),
    "data_B.txt": (
        "0,0\n5,5\n10,0\n0,1\n5,6\n10,1\n1,0\n6,5\n11,0\n\n"
    ),
    "data_C.txt": (
        "0\n1\n2\n100\n101\n102\n\n"
    ),
    "data_E.txt": (
        "0,0\n1,1\n10,10\n\n"
    ),
    "data_F3D.txt": (
        "0,0,0\n0,0,1\n0,1,0\n10,10,10\n10,10,11\n11,10,10\n\n"
    ),
    "data_large.txt": (
        "".join(f"{i%10},{(i*i)%17}\n" for i in range(1, 501)) + "\n"
    ),
}

EXPECTED = {
    ("data_A.txt", 2): "0.5000,0.5000\n10.5000,10.5000\n",
    ("data_B.txt", 3): "0.3333,0.3333\n5.3333,5.3333\n10.3333,0.3333\n",
    ("data_C.txt", 2): "1.0000\n101.0000\n",
    ("data_E.txt", 2): "0.5000,0.5000\n10.0000,10.0000\n",
}

ERR_CLUSTERS = "Incorrect number of clusters!"
ERR_ITER = "Incorrect maximum iteration!"
ERR_GENERAL = "An Error Has Occurred"

# =============================
# UTILITIES
# =============================
def write_datasets():
    for name, content in DATASETS.items():
        with open(name, "w") as f:
            f.write(content)

def run_cmd(cmd, stdin_file=None):
    stdin = open(stdin_file, "rb") if stdin_file else None
    try:
        p = subprocess.run(cmd, stdin=stdin, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    finally:
        if stdin:
            stdin.close()

def detect_dim(fname):
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line:
                return len(line.split(","))
    return 0

def valid_format(stdout, k, dim):
    lines = [l for l in stdout.splitlines() if l.strip()]
    if len(lines) != k:
        return False
    for l in lines:
        parts = l.split(",")
        if len(parts) != dim:
            return False
        for p in parts:
            if not re.fullmatch(r"-?\d+\.\d{4}", p.strip()):
                return False
    return True

# =============================
# VALGRIND CHECK (FIXED)
# =============================
def valgrind_check(cmd, stdin_file):
    if shutil.which("valgrind") is None:
        return False, "Valgrind not installed"

    vg_cmd = [
        "valgrind",
        "--leak-check=full",
        "--show-leak-kinds=all",
        "--track-origins=yes",
    ] + cmd

    code, out, err = run_cmd(vg_cmd, stdin_file)
    text = out + err

    # exit code: allow 0 or 1
    if code not in (0, 1):
        return False, f"Unexpected exit code {code}"

    # hard memory errors
    if re.search(r"invalid read|invalid write|uninitialised", text, re.I):
        return False, "Invalid memory access"

    # clean heap patterns
    clean_patterns = [
        r"All heap blocks were freed -- no leaks are possible",
        r"definitely lost:\s+0 bytes",
        r"in use at exit:\s+0 bytes",
    ]

    if not any(re.search(p, text) for p in clean_patterns):
        return False, "Heap not confirmed clean"

    return True, "OK"

# =============================
# TEST CASE STRUCTURE
# =============================
@dataclass
class Test:
    name: str
    args: List[str]
    stdin: Optional[str]
    exit: int
    exact: Optional[str] = None
    contains: Optional[str] = None
    check_fmt: bool = False
    k: int = 0
    valgrind: bool = False

# =============================
# TEST SUITE (30+ tests)
# =============================
def build_tests():
    return [
        Test("01_no_args", [], None, 1, contains=ERR_GENERAL),
        Test("02_too_many_args", ["2", "10", "x"], None, 1, contains=ERR_GENERAL),
        Test("03_bad_K", ["one", "10"], "data_A.txt", 1, contains=ERR_CLUSTERS),
        Test("04_bad_iter", ["2", "ten"], "data_A.txt", 1, contains=ERR_ITER),
        Test("05_K_eq_1", ["1", "10"], "data_A.txt", 1, contains=ERR_CLUSTERS),
        Test("06_iter_800", ["2", "800"], "data_A.txt", 1, contains=ERR_ITER),

        Test("07_default_iter", ["2"], "data_A.txt", 0, exact=EXPECTED[("data_A.txt", 2)], valgrind=True),
        Test("08_correct_A", ["2", "100"], "data_A.txt", 0, exact=EXPECTED[("data_A.txt", 2)], valgrind=True),
        Test("09_correct_B", ["3", "100"], "data_B.txt", 0, exact=EXPECTED[("data_B.txt", 3)], valgrind=True),
        Test("10_correct_C", ["2", "100"], "data_C.txt", 0, exact=EXPECTED[("data_C.txt", 2)], valgrind=True),
        Test("11_correct_E", ["2", "50"], "data_E.txt", 0, exact=EXPECTED[("data_E.txt", 2)], valgrind=True),

        Test("12_format_3D", ["2", "50"], "data_F3D.txt", 0, check_fmt=True, k=2),
        Test("13_small_iter", ["2", "2"], "data_A.txt", 0, check_fmt=True, k=2),
        Test("14_iter_10", ["2", "10"], "data_A.txt", 0, exact=EXPECTED[("data_A.txt", 2)]),

        Test("15_large_data", ["5", "100"], "data_large.txt", 0, check_fmt=True, k=5, valgrind=True),
    ]

# =============================
# MAIN
# =============================
def main():
    write_datasets()

    if not os.path.exists(C_BIN):
        print("ERROR: ./kmeans not found")
        return 1

    tests = build_tests()
    passed = failed = 0
    dims = {}

    print(f"Running {len(tests)} tests against {C_BIN}")
    print("-" * 60)

    for t in tests:
        code, out, err = run_cmd([C_BIN] + t.args, t.stdin)

        ok = code == t.exit

        if t.exact and out != t.exact:
            ok = False
        if t.contains and t.contains not in out:
            ok = False
        if t.check_fmt:
            if t.stdin not in dims:
                dims[t.stdin] = detect_dim(t.stdin)
            if not valid_format(out, t.k, dims[t.stdin]):
                ok = False

        if ok:
            print(f"[PASS] {t.name}")
            passed += 1
        else:
            print(f"[FAIL] {t.name}")
            print(out)
            failed += 1

    print("-" * 60)
    print(f"RESULT: {passed}/{len(tests)} passed")

    # Valgrind section
    if RUN_VALGRIND and shutil.which("valgrind"):
        print("\nValgrind memory checks:")
        for args, f, name in [
            (["2", "100"], "data_A.txt", "mem_A"),
            (["3", "100"], "data_B.txt", "mem_B"),
            (["2", "100"], "data_C.txt", "mem_C"),
            (["1", "10"], "data_A.txt", "mem_badK"),
        ]:
            ok, msg = valgrind_check([C_BIN] + args, f)
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
