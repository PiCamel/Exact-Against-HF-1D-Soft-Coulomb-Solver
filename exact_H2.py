# Exact two-electron solver for the H2 molecule: two nuclei of charge 1 at x = -R/2 and +R/2.
# Same N x N grid method as exact_H_minus.py, only the external potential changes.
# The nucleus-nucleus repulsion w(R) is a constant at fixed nuclei (Born-Oppenheimer): it does not change Psi,
# but it must be added to get the total energy of the molecule.

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from common import make_grid, w, v_2atom, H_two_electron, A_SOFT


def run(R=3.5, L=10.0, h=0.1, a=A_SOFT):
    x, N = make_grid(L, h)    # odd N, x = 0 on the grid

    E,P = spla.eigsh(H_two_electron(x, h, v_2atom(x,1,a,R), a),k=1,which = 'SA',tol = 1e-10)
    # so the explanation here i htink is in that after solving the system stays in the lowest eigen value pair , basically meaning the system settles into the lowest energy state
    Psi = P[:, 0].reshape(N, N) / h        # flat vector -> N x N table, Psi[i, j] = Psi(x_i, x_j)
    if Psi.sum() < 0:                      # eigenvector sign is arbitrary
        Psi = -Psi
    n = 2 * (Psi**2).sum(axis=1) * h       # n(x_i) = 2 * sum_j |Psi_ij|^2 * h

    E_nn = w(R, a)                         # nucleus-nucleus repulsion
    return {"E": E[0] + E_nn, "E_elec": E[0], "E_nn": E_nn, "n": n, "x": x, "Psi": Psi, "h": h, "N": N, "R": R}


if __name__ == "__main__":
    res = run(R=3.5)
    print("exact H2 total energy:", res["E"], "(electronic", res["E_elec"], "+ nuclear", res["E_nn"], ")")
    print("integral of n        :", res["h"] * np.sum(res["n"]))   # must be 2
    plt.plot(res["x"], res["n"]); plt.xlabel("x (bohr)"); plt.ylabel("n(x)")
    plt.title("Exact density, H2 (R = 3.5)")
    plt.show()
