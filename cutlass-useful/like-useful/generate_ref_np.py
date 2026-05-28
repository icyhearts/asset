"""
Generate reference A, B, C matrices for sgemm_1_ref_np.cu verification.

Usage: python generate_ref_np.py m n k [transA] [transB]
  - Args order matches sgemm_1_ref_np.cu command line
  - Default: transA='N', transB='T'

Memory layout convention:
  CuTe/BLAS uses column-major. NumPy uses row-major (C order).
  Key insight: row-major (k, m) array has identical byte layout to column-major (m, k).
  So we create numpy arrays with transposed shapes and use tofile() directly.
"""

import numpy as np
import sys

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} m n k [transA] [transB]")
        print(f"  Default: transA='N', transB='T'")
        sys.exit(1)

    m = int(sys.argv[1])
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    transA = sys.argv[4] if len(sys.argv) > 4 else 'N'
    transB = sys.argv[5] if len(sys.argv) > 5 else 'T'

    print(f"M={m}, N={n}, K={k}, transA={transA}, transB={transB}")

    # Generate A matrix
    # transA='N': CuTe expects col-major (m,k) with stride (1,m)
    #   -> numpy row-major (k,m) has same byte layout
    #   -> A_logical (for matmul) = np_A.T, shape (m,k)
    # transA='T': CuTe expects row-major (m,k) with stride (k,1)
    #   -> numpy row-major (m,k) matches directly
    #   -> A_logical = np_A, shape (m,k)
    if transA == 'N':
        np_A = np.random.uniform(-1, 1, (k, m)).astype(np.float32)
        A_logical = np_A.T  # (m, k) view
        print(f"A: numpy shape ({k},{m}) row-major = CuTe col-major ({m},{k})")
    else:
        np_A = np.random.uniform(-1, 1, (m, k)).astype(np.float32)
        A_logical = np_A    # (m, k) directly
        print(f"A: numpy shape ({m},{k}) row-major = CuTe row-major ({m},{k})")

    # Generate B matrix
    # transB='T': CuTe expects col-major (n,k) with stride (1,n)
    #   -> numpy row-major (k,n) has same byte layout
    #   -> B_logical = np_B.T, shape (n,k)
    # transB='N': CuTe expects row-major (n,k) with stride (k,1)
    #   -> numpy row-major (n,k) matches directly
    #   -> B_logical = np_B, shape (n,k)
    if transB == 'T':
        np_B = np.random.uniform(-1, 1, (k, n)).astype(np.float32)
        B_logical = np_B.T  # (n, k) view
        print(f"B: numpy shape ({k},{n}) row-major = CuTe col-major ({n},{k})")
    else:
        np_B = np.random.uniform(-1, 1, (n, k)).astype(np.float32)
        B_logical = np_B    # (n, k) directly
        print(f"B: numpy shape ({n},{k}) row-major = CuTe row-major ({n},{k})")

    # Compute C = A_logical(m,k) @ B_logical(n,k)^T = A_logical @ B_logical.T -> (m,n)
    C_logical = (A_logical @ B_logical.T).astype(np.float32)  # (m, n)
    print(f"C_logical: shape {C_logical.shape}")

    # C is always col-major (m,n) with stride (1,m) in CuTe
    # -> store as numpy row-major (n,m)
    np_C = np.ascontiguousarray(C_logical.T)  # (n, m) contiguous row-major
    print(f"C: numpy shape ({n},{m}) row-major = CuTe col-major ({m},{n})")

    # Save binary files
    np_A.tofile("A.npy")
    np_B.tofile("B.npy")
    np_C.tofile("C.npy")

    print(f"Saved A.npy ({np_A.nbytes} bytes), B.npy ({np_B.nbytes} bytes), C.npy ({np_C.nbytes} bytes)")

if __name__ == "__main__":
    main()
