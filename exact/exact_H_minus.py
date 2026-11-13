# Exact two-electron solver for a single nucleus of charge Z (default Z = 1, H-). He (Z = 2) has its own file, exact_He.py.
# The two-electron wavefunction Psi(x1, x2) lives on an N x N grid, i.e. N^2 unknowns.
# This is the benchmark every approximate method is compared against.

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) # so common.py (one folder up) can be imported
from common import make_grid, v_1atom, H_two_electron, A_SOFT


def run(Z=1, L=10.0, h=0.1, a=A_SOFT):
    x, N = make_grid(L, h)    # odd N, x = 0 on the grid

    E,P = spla.eigsh(H_two_electron(x, h, v_1atom(x,Z,a), a),k=1,which = 'SA',tol = 1e-10)
    # so the explanation here i htink is in that after solving the system stays in the lowest eigen value pair , basically meaning the system settles into the lowest energy state
    Psi = P[:, 0].reshape(N, N) / h        # flat vector -> N x N table, Psi[i, j] = Psi(x_i, x_j)
    if Psi.sum() < 0:                      # eigenvector sign is arbitrary
        Psi = -Psi
    n = 2 * (Psi**2).sum(axis=1) * h       # n(x_i) = 2 * sum_j |Psi_ij|^2 * h

    # one nucleus, so there is no nucleus-nucleus repulsion: electronic energy = total energy
    return {"E": E[0], "n": n, "x": x, "Psi": Psi, "h": h, "N": N}


if __name__ == "__main__":
    res = run(Z=1)
    print("exact H- energy      :", res["E"])
    print("integral of n        :", res["h"] * np.sum(res["n"]))   # must be 2
    plt.plot(res["x"], res["n"]); plt.xlabel("x (bohr)"); plt.ylabel("n(x)")
    plt.title("Exact density, H-")
    plt.show()
