import subprocess
import random
import math
import os
import sys

# --- Constants and Configuration ---
C_EXEC = "./kmeans"
PY_SCRIPT = "kmeans.py"
INPUT_FILE = "input_test_data.txt"
EPSILON = 0.001

# Terminal colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def generate_input(filename, N, dim):
    """Generates random input data and writes it to a file."""
    points = []
    with open(filename, 'w') as f:
        for _ in range(N):
            # Wide range to encourage centroid movement
            vec = [round(random.uniform(-20, 20), 4) for _ in range(dim)]
            points.append(vec)
            f.write(",".join(map(str, vec)) + "\n")
    return points

def dist(v1, v2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(v1, v2)))

def run_internal_logic(points, K, max_iter):
    """
    Reference implementation of the K-means algorithm for comparison.
    Includes empty-cluster logic: copy the first input vector as fallback.
    """
    if K >= len(points): return None  # Expected error case

    centroids = [p[:] for p in points[:K]]

    for _ in range(max_iter):
        clusters = [[] for _ in range(K)]

        # Assignment step
        for p in points:
            distances = [dist(p, c) for c in centroids]
            min_d = min(distances)
            closest = distances.index(min_d)
            clusters[closest].append(p)

        new_centroids = []
        changed = False

        # Update step
        for i in range(K):
            cluster = clusters[i]
            if cluster:
                # Compute mean
                dim = len(cluster[0])
                avg = [0.0] * dim
                for p in cluster:
                    for d in range(dim):
                        avg[d] += p[d]
                new_c = [x / len(cluster) for x in avg]
                new_centroids.append(new_c)
            else:
                # Empty cluster: fall back to first input vector
                new_centroids.append(points[0][:])

        # Convergence check
        for i in range(K):
            if dist(centroids[i], new_centroids[i]) >= EPSILON:
                changed = True

        centroids = new_centroids[:]
        if not changed:
            break

    return centroids

def run_program(cmd_args, input_file):
    """Runs an external program (C or Python) and returns its output."""
    try:
        with open(input_file, 'r') as infile:
            res = subprocess.run(
                cmd_args,
                stdin=infile,
                capture_output=True,
                text=True,
                timeout=5  # Timeout to prevent infinite loops
            )
        return res.stdout.strip(), res.stderr, res.returncode
    except subprocess.TimeoutExpired:
        return None, "Timeout", 1
    except Exception as e:
        return None, str(e), 1

def parse_centroids(output):
    """Converts text output to a list of vectors."""
    if not output: return []
    res = []
    for line in output.split('\n'):
        try:
            vec = [float(x) for x in line.split(',')]
            res.append(vec)
        except:
            pass
    return res

def compare_vectors(vecs1, vecs2, tolerance=0.0005):
    """Compares two lists of vectors within a given tolerance."""
    if len(vecs1) != len(vecs2): return False
    for v1, v2 in zip(vecs1, vecs2):
        if len(v1) != len(v2): return False
        if dist(v1, v2) > tolerance: return False
    return True

def run_test(test_name, K, iter_val, N, dim, expect_error=None):
    print(f"Running {test_name} [K={K}, iter={iter_val}, N={N}, dim={dim}]...", end=" ")

    # 1. Generate input
    points = generate_input(INPUT_FILE, N, dim)

    # Build argument lists
    args_c = [C_EXEC, str(K)]
    args_py = ["python3", PY_SCRIPT, str(K)]
    if iter_val is not None:
        args_c.append(str(iter_val))
        args_py.append(str(iter_val))

    max_iter_internal = iter_val if iter_val else 400

    # 2. Run C implementation
    out_c, err_c, code_c = run_program(args_c, INPUT_FILE)

    # 3. Run Python implementation
    out_py, err_py, code_py = run_program(args_py, INPUT_FILE)

    # 4. Check expected error cases
    if expect_error:
        success = True
        if out_c != expect_error:
            print(f"\n{RED}[C Failed]{RESET} Expected error '{expect_error}', got: '{out_c}'")
            success = False
        if out_py != expect_error:
            print(f"\n{RED}[Py Failed]{RESET} Expected error '{expect_error}', got: '{out_py}'")
            success = False
        if success: print(f"{GREEN}PASSED (Errors matched){RESET}")
        else: print(f"{RED}FAILED{RESET}")
        return

    # 5. Check for crashes
    if code_c != 0 or code_py != 0:
        print(f"\n{RED}CRASHED{RESET}")
        if code_c != 0: print(f"C Error: {out_c}")
        if code_py != 0: print(f"Py Error: {out_py}")
        return

    # 6. Compute reference result
    ref_centroids = run_internal_logic(points, K, max_iter_internal)

    # 7. Compare outputs
    cents_c = parse_centroids(out_c)
    cents_py = parse_centroids(out_py)

    c_ok = compare_vectors(ref_centroids, cents_c)
    py_ok = compare_vectors(ref_centroids, cents_py)
    cross_ok = compare_vectors(cents_c, cents_py)

    if c_ok and py_ok and cross_ok:
        print(f"{GREEN}PASSED ✅{RESET}")
    else:
        print(f"{RED}FAILED ❌{RESET}")
        if not c_ok: print(f"  - C output differs from reference.")
        if not py_ok: print(f"  - Python output differs from reference.")
        if not cross_ok: print(f"  - C and Python outputs differ from each other.")

        # Print first centroid for debugging
        print(f"  Ref first: {ref_centroids[0]}")
        if cents_c: print(f"  C   first: {cents_c[0]}")
        if cents_py: print(f"  Py  first: {cents_py[0]}")

def main():
    # Compile C code
    print("--- Compiling C Code ---")
    ret = os.system("gcc -ansi -Wall -Wextra -Werror -pedantic-errors kmeans.c -o kmeans -lm")
    if ret != 0:
        print(f"{RED}Compilation Failed! Fix C errors first.{RESET}")
        return

    print("\n--- Starting Tests ---")

    # Test 1: Standard run
    run_test("Standard", K=3, iter_val=100, N=50, dim=2)

    # Test 2: Default iter (400)
    run_test("Default Iter", K=2, iter_val=None, N=40, dim=3)

    # Test 3: High dimensionality
    run_test("High Dim", K=5, iter_val=50, N=100, dim=10)

    # Test 4: High K (still valid)
    run_test("High K", K=10, iter_val=100, N=50, dim=2)

    # --- Edge cases and error handling ---

    # Test 5: K=1 (invalid)
    run_test("Invalid K=1", K=1, iter_val=100, N=50, dim=2, expect_error="Incorrect number of clusters!")

    # Test 6: K >= N (invalid)
    run_test("Invalid K>=N", K=20, iter_val=100, N=10, dim=2, expect_error="Incorrect number of clusters!")

    # Test 7: iter too large
    run_test("Invalid Iter > 800", K=3, iter_val=1000, N=50, dim=2, expect_error="Incorrect maximum iteration!")

    # Test 8: iter too small (C truncates floats, so 1.5 becomes 1)
    run_test("Invalid Iter <= 1", K=3, iter_val=1, N=50, dim=2, expect_error="Incorrect maximum iteration!")

    # Cleanup
    if os.path.exists(INPUT_FILE):
        os.remove(INPUT_FILE)
    print("\nDone.")

if __name__ == "__main__":
    main()
