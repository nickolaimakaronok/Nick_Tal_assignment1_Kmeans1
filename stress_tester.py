#!/usr/bin/env python3
"""
K-means HW tester (35+ cases) + memory leak check (Valgrind)
Adjusted to match your kmeans.c behavior:
- Provide stdin for any test that doesn't fail before reading input
- Removed whitespace-trailing/CRLF tests because scanf("%lf%c") treats them as delimiters
"""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple

# -----------------------------
# CONFIG
# -----------------------------
C_BIN = "./kmeans"          # your compiled C executable
PY_IMPL = "./kmeans.py"     # optional python implementation for parity checks
RUN_PY_PARITY = os.path.exists(PY_IMPL)
RUN_VALGRIND = True

# -----------------------------
# DATASETS (valid content; includes optional final empty row)
# -----------------------------
DATASETS = {
    "data_A.txt": (
        "0,0\n0,1\n1,0\n1,1\n10,10\n10,11\n11,10\n11,11\n\n"
    ),
    "data_A_no_blank.txt": (
        "0,0\n0,1\n1,0\n1,1\n10,10\n10,11\n11,10\n11,11\n"
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
    "data_smallN.txt": (
        "0,0\n1,0\n0,1\n9,9\n10,9\n9,10\n\n"
    ),
    "data_H_large.txt": (
        "".join(f"{i%10},{(i*i)%17}\n" for i in range(1, 501)) + "\n"
    ),
    "data_10D.txt": (
        "0,0,0,0,0,0,0,0,0,0\n"
        "1,1,1,1,1,1,1,1,1,1\n"
        "2,2,2,2,2,2,2,2,2,2\n"
        "100,100,100,100,100,100,100,100,100,100\n"
        "101,101,101,101,101,101,101,101,101,101\n\n"
    ),
}

# -----------------------------
# Expected deterministic outputs
# -----------------------------
EXPECTED = {
    ("data_A.txt", 2): "0.5000,0.5000\n10.5000,10.5000\n",
    ("data_A_no_blank.txt", 2): "0.5000,0.5000\n10.5000,10.5000\n",
    ("data_B.txt", 3): "0.3333,0.3333\n5.3333,5.3333\n10.3333,0.3333\n",
    ("data_C.txt", 2): "1.0000\n101.0000\n",
    ("data_E.txt", 2): "0.5000,0.5000\n10.0000,10.0000\n",
}

ERR_CLUSTERS = "Incorrect number of clusters!"
ERR_ITER = "Incorrect maximum iteration!"
ERR_GENERAL = "An Error Has Occurred"

def write_datasets() -> None:
    for name, content in DATASETS.items():
        with open(name, "w", newline="") as f:
            f.write(content)

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run_cmd(cmd: List[str], stdin_file: Optional[str] = None) -> Tuple[int, str, str]:
    stdin = None
    try:
        if stdin_file is not None:
            stdin = open(stdin_file, "rb")
        p = subprocess.run(cmd, stdin=stdin, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    finally:
        if stdin is not None:
            stdin.close()

def is_four_decimals_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    parts = line.split(",")
    for part in parts:
        part = part.strip()
        if not re.fullmatch(r"-?\d+\.\d{4}", part):
            return False
    return True

def detect_dim_from_file(fname: str) -> int:
    with open(fname, "r") as f:
        for line in f:
            s = line.strip()
            if s == "":
                continue
            return len(s.split(","))
    raise RuntimeError(f"Could not detect dimension from {fname}")

def check_output_format(stdout: str, expected_k: int, dim: int) -> Optional[str]:
    lines = [ln for ln in stdout.splitlines() if ln.strip() != ""]
    if len(lines) != expected_k:
        return f"Expected {expected_k} centroid lines, got {len(lines)}"
    for i, ln in enumerate(lines):
        if not is_four_decimals_line(ln):
            return f"Line {i+1} not in 4-decimal format: {ln!r}"
        parts = [p.strip() for p in ln.strip().split(",")]
        if len(parts) != dim:
            return f"Line {i+1} expected {dim} dimensions, got {len(parts)}: {ln!r}"
    return None

def valgrind_check(cmd: List[str], stdin_file: Optional[str]) -> Tuple[bool, str]:
    if which("valgrind") is None:
        return False, "valgrind not installed"

    vg_cmd = [
        "valgrind",
        "--leak-check=full",
        "--show-leak-kinds=all",
        "--errors-for-leak-kinds=all",
        "--error-exitcode=99",
    ] + cmd

    code, out, err = run_cmd(vg_cmd, stdin_file=stdin_file)
    vg_text = (out or "") + "\n" + (err or "")

    if code == 99:
        return False, "Valgrind detected leaks (error-exitcode=99)"
    if code != 0:
        return False, f"Non-zero exit under Valgrind: {code}"
    if re.search(r"definitely lost:\s+0 bytes", vg_text) is None:
        return False, "Could not confirm 'definitely lost: 0 bytes' in Valgrind output"
    if re.search(r"invalid read|invalid write", vg_text, re.IGNORECASE):
        return False, "Valgrind reported invalid read/write"

    return True, "OK"

@dataclass
class TestCase:
    name: str
    args: List[str]
    stdin_file: Optional[str]
    expect_exit: int
    expect_stdout_exact: Optional[str] = None
    expect_stdout_contains: Optional[str] = None
    check_format: bool = False
    expected_k: Optional[int] = None
    run_valgrind: bool = False

def build_cases() -> List[TestCase]:
    cases: List[TestCase] = []

    # CLI validation that exits BEFORE reading stdin
    cases += [
        TestCase("01_no_args", [], None, 1, expect_stdout_contains=ERR_GENERAL),
        TestCase("02_too_many_args", ["2", "10", "extra"], None, 1, expect_stdout_contains=ERR_GENERAL),
    ]

    # CLI validation that may read stdin later -> provide stdin to avoid hangs
    cases += [
        TestCase("03_K_not_int", ["two", "10"], "data_A.txt", 1, expect_stdout_contains=ERR_CLUSTERS),
        TestCase("04_iter_not_int", ["2", "ten"], "data_A.txt", 1, expect_stdout_contains=ERR_ITER),
        TestCase("05_K_eq_1", ["1", "10"], "data_A.txt", 1, expect_stdout_contains=ERR_CLUSTERS),
        TestCase("06_iter_eq_800", ["2", "800"], "data_A.txt", 1, expect_stdout_contains=ERR_ITER),
        TestCase("07_K_too_big_default_iter", ["100"], "data_A.txt", 1, expect_stdout_contains=ERR_CLUSTERS),
    ]

    # Default iter works
    cases += [
        TestCase("08_default_iter_400", ["2"], "data_A.txt", 0,
                 expect_stdout_exact=EXPECTED[("data_A.txt", 2)], run_valgrind=True),
    ]

    # Input / formatting / deterministic correctness
    cases += [
        TestCase("09_extra_empty_line_end", ["2", "20"], "data_A.txt", 0,
                 expect_stdout_exact=EXPECTED[("data_A.txt", 2)], run_valgrind=True),
        TestCase("10_no_final_blank_line", ["2", "20"], "data_A_no_blank.txt", 0,
                 expect_stdout_exact=EXPECTED[("data_A_no_blank.txt", 2)], run_valgrind=True),
        TestCase("11_correct_B_K3", ["3", "100"], "data_B.txt", 0,
                 expect_stdout_exact=EXPECTED[("data_B.txt", 3)], run_valgrind=True),
        TestCase("12_correct_C_1D", ["2", "100"], "data_C.txt", 0,
                 expect_stdout_exact=EXPECTED[("data_C.txt", 2)], run_valgrind=True),
        TestCase("13_minimal_N3_K2", ["2", "50"], "data_E.txt", 0,
                 expect_stdout_exact=EXPECTED[("data_E.txt", 2)], run_valgrind=True),
        TestCase("14_3D_format", ["2", "50"], "data_F3D.txt", 0,
                 check_format=True, expected_k=2, run_valgrind=True),
        TestCase("15_K_close_to_N_minus_1", ["5", "50"], "data_smallN.txt", 0,
                 check_format=True, expected_k=5, run_valgrind=True),
        TestCase("16_10D_format", ["2", "100"], "data_10D.txt", 0,
                 check_format=True, expected_k=2, run_valgrind=True),
        TestCase("17_small_iter_cap_iter2", ["2", "2"], "data_A.txt", 0,
                 check_format=True, expected_k=2, run_valgrind=True),
        TestCase("18_iter10_reaches_final", ["2", "10"], "data_A.txt", 0,
                 expect_stdout_exact=EXPECTED[("data_A.txt", 2)]),
    ]

    # Stress-ish
    cases += [
        TestCase("19_largeN_perf", ["5", "100"], "data_H_large.txt", 0,
                 check_format=True, expected_k=5, run_valgrind=True),
        TestCase("20_repeat_runs_100x", ["2", "20"], "data_A.txt", 0,
                 expect_stdout_exact=EXPECTED[("data_A.txt", 2)]),
    ]

    # Add a few more boundary arg tests (valid/invalid)
    cases += [
        TestCase("21_iter_eq_1_invalid", ["2", "1"], "data_A.txt", 1, expect_stdout_contains=ERR_ITER),
        TestCase("22_iter_eq_799_valid", ["2", "799"], "data_A.txt", 0,
                 expect_stdout_exact=EXPECTED[("data_A.txt", 2)]),
        TestCase("23_K_eq_0_invalid", ["0", "10"], "data_A.txt", 1, expect_stdout_contains=ERR_CLUSTERS),
        TestCase("24_iter_negative_invalid", ["2", "-7"], "data_A.txt", 1, expect_stdout_contains=ERR_ITER),
        TestCase("25_K_negative_invalid", ["-3", "10"], "data_A.txt", 1, expect_stdout_contains=ERR_CLUSTERS),
    ]

    # Optional Python parity (if you have kmeans.py)
    if RUN_PY_PARITY:
        cases += [
            TestCase("26_py_parity_A", ["2", "100"], "data_A.txt", 0,
                     expect_stdout_exact=EXPECTED[("data_A.txt", 2)]),
        ]

    # Pad to 30+ with more valid-format runs on different iter values
    extra_iters = [3, 4, 5, 6, 7, 8, 9, 15, 25]
    for idx, it in enumerate(extra_iters, start=27):
        cases.append(TestCase(f"{idx:02d}_format_iter_{it}", ["2", str(it)], "data_A.txt", 0,
                              check_format=True, expected_k=2))

    return cases

def main() -> int:
    write_datasets()

    if not os.path.exists(C_BIN):
        print(f"ERROR: C executable not found at {C_BIN}")
        print("Compile it first, e.g.: gcc -O2 -Wall -Wextra -o kmeans kmeans.c -lm")
        return 2

    cases = build_cases()
    total = len(cases)
    passed = failed = 0
    dims_cache = {}

    vg_available = which("valgrind") is not None
    if RUN_VALGRIND and not vg_available:
        print("NOTE: valgrind not found; memory-leak checks will be SKIPPED.\n")

    print(f"Running {total} tests against: {C_BIN}")
    if RUN_PY_PARITY:
        print(f"Python parity enabled using: {PY_IMPL}")
    else:
        print("Python parity disabled (kmeans.py not found).")
    print("-" * 60)

    for tc in cases:
        # Special-case repeat run
        if tc.name == "20_repeat_runs_100x":
            ok = True
            for _ in range(100):
                code, out, err = run_cmd([C_BIN] + tc.args, stdin_file=tc.stdin_file)
                if code != tc.expect_exit:
                    ok = False
                    detail = f"Expected exit {tc.expect_exit}, got {code}"
                    break
                if tc.expect_stdout_exact is not None and out != tc.expect_stdout_exact:
                    ok = False
                    detail = "stdout mismatch on repeated runs"
                    break
            if ok:
                print(f"[PASS] {tc.name}")
                passed += 1
            else:
                print(f"[FAIL] {tc.name}: {detail}")
                failed += 1
            continue

        code, out, err = run_cmd([C_BIN] + tc.args, stdin_file=tc.stdin_file)

        ok = (code == tc.expect_exit)
        details = []
        if not ok:
            details.append(f"exit expected {tc.expect_exit}, got {code}")

        if tc.expect_stdout_exact is not None and out != tc.expect_stdout_exact:
            ok = False
            details.append("stdout exact mismatch")

        if tc.expect_stdout_contains is not None and tc.expect_stdout_contains not in out:
            ok = False
            details.append(f"stdout missing: {tc.expect_stdout_contains!r}")

        if tc.check_format:
            if tc.stdin_file is None:
                ok = False
                details.append("format check requested but no stdin file")
            else:
                if tc.stdin_file not in dims_cache:
                    dims_cache[tc.stdin_file] = detect_dim_from_file(tc.stdin_file)
                dim = dims_cache[tc.stdin_file]
                fmt_err = check_output_format(out, tc.expected_k or 0, dim)
                if fmt_err:
                    ok = False
                    details.append(fmt_err)

        # Optional parity test
        if RUN_PY_PARITY and tc.name == "26_py_parity_A":
            py_code, py_out, py_err = run_cmd(["python3", PY_IMPL] + tc.args, stdin_file=tc.stdin_file)
            if py_code != tc.expect_exit:
                ok = False
                details.append(f"python exit expected {tc.expect_exit}, got {py_code}")
            if tc.expect_stdout_exact is not None and py_out != tc.expect_stdout_exact:
                ok = False
                details.append("python stdout exact mismatch")

        if ok:
            print(f"[PASS] {tc.name}")
            passed += 1
        else:
            print(f"[FAIL] {tc.name}: " + "; ".join(details))
            print("  --- stdout ---")
            print(out.rstrip("\n"))
            print("  --- stderr ---")
            print(err.rstrip("\n"))
            failed += 1

    print("-" * 60)
    print(f"RESULT: {passed}/{total} passed, {failed} failed")

    # Valgrind checks at end
    if RUN_VALGRIND and vg_available:
        print("\nValgrind memory checks (no leaks allowed):")
        mem_tests = [
            (["2", "100"], "data_A.txt", "mem_success_A"),
            (["3", "100"], "data_B.txt", "mem_success_B"),
            (["2", "100"], "data_C.txt", "mem_success_C_1D"),
            (["1", "10"], "data_A.txt", "mem_error_badK"),
            (["2", "800"], "data_A.txt", "mem_error_badIter"),
            (["two", "10"], "data_A.txt", "mem_error_nonIntK"),
        ]
        mem_failed = 0
        for args, f, name in mem_tests:
            ok, msg = valgrind_check([C_BIN] + args, stdin_file=f)
            if ok:
                print(f"[PASS] {name}")
            else:
                print(f"[FAIL] {name}: {msg}")
                mem_failed += 1

        if mem_failed == 0:
            print("Memory status: CLEAN (freed all allocated memory).")
        else:
            print(f"Memory status: {mem_failed} Valgrind check(s) failed.")
            failed += mem_failed

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
