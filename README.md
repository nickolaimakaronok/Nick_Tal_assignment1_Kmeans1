# K-Means Clustering

An implementation of the K-Means clustering algorithm in both **C** and **Python**, with cross-language output parity and a comprehensive test suite.

## Overview

K-Means partitions N data points into K clusters by iteratively:
1. **Assigning** each point to its nearest centroid (Euclidean distance)
2. **Updating** each centroid to the mean of its assigned points
3. **Repeating** until convergence (Δ < ε = 0.001) or max iterations reached

Initial centroids are the first K points from the input.

---

## Project Structure

```
kmeans-clustering/
├── kmeans.c              # C implementation (linked list, ANSI C)
├── kmeans.py             # Python implementation
├── tester.py             # Cross-language correctness tester (C vs Python vs reference)
├── kmeans_tester.py      # Extended C tester with 30+ cases
├── stress_tester.py      # Stress tester with Valgrind memory checks
└── data/
    ├── input_1.txt
    ├── input_2.txt
    ├── input_3.txt
    ├── data_A.txt
    └── ...
```

---

## How to Build & Run

### C Implementation

```bash
gcc -ansi -Wall -Wextra -Werror -pedantic-errors kmeans.c -o kmeans -lm
./kmeans <K> [max_iter] < input.txt
```

### Python Implementation

```bash
python3 kmeans.py <K> [max_iter] < input.txt
```

### Arguments

| Argument   | Required | Description                          | Valid range  |
|------------|----------|--------------------------------------|--------------|
| `K`        | Yes      | Number of clusters                   | 1 < K < N    |
| `max_iter` | No       | Maximum iterations (default: 400)    | 1 < iter < 800 |

Input is read from **stdin** — one data point per line, coordinates separated by commas.

---

## Example

```bash
echo -e "0,0\n0,1\n1,0\n1,1\n10,10\n10,11\n11,10\n11,11" | ./kmeans 2 100
```

**Output:**
```
0.5000,0.5000
10.5000,10.5000
```

---

## Error Handling

| Error message                    | Cause                              |
|----------------------------------|------------------------------------|
| `Incorrect number of clusters!`  | K ≤ 1 or K ≥ N or non-integer K   |
| `Incorrect maximum iteration!`   | iter ≤ 1 or iter ≥ 800            |
| `An Error Has Occurred`          | Wrong number of arguments or I/O   |

All errors exit with code `1`.

---

## Testing

### Cross-language tester (C vs Python vs reference)
```bash
python3 tester.py
```

### Extended C tester (30+ cases)
```bash
python3 kmeans_tester.py
```

### Stress tester with Valgrind
```bash
python3 stress_tester.py
```

---

## Implementation Details

**C** — uses linked lists (`struct vector`, `struct cord`) for dynamic dimensionality. Memory is fully freed after each run (verified with Valgrind).

**Python** — list-based implementation matching the C behavior exactly, including the empty-cluster fallback (centroid reset to first input vector).

**Convergence** — stops when all centroid movements are below ε = 0.001, or when `max_iter` is reached.

---

## Time Complexity

| Step        | Complexity      |
|-------------|-----------------|
| Assignment  | O(N × K × d)    |
| Update      | O(N × d)        |
| Full run    | O(T × N × K × d) |

Where T = iterations, N = points, K = clusters, d = dimensions.

---

## Tech Stack

- **C** (ANSI C99) — gcc with `-Wall -Wextra -Werror`
- **Python 3** — no external libraries
- **Valgrind** — memory leak verification
