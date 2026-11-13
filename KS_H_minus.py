# Kohn-Sham solver for a single nucleus of charge Z (Z = 1 is H-, Z = 2 is He).
# For 2 electrons in one orbital, KS with exact exchange (v_x = -v_H/2, v_c = 0) is identical to restricted Hartree-Fock.
# The orbital lives on an N-point grid, i.e. N unknowns instead of N^2.

import numpy as np
import scipy as s
import matplotlib.pyplot as plt
from common import make_grid, v_ext, T_1e, V_H, one_electron


def run(Z=1, L=10.0, h=0.1, alpha=0.4, err_thres=1e-6, max_iter=1000):
  h_KS = h
  x_KS, N_KS = make_grid(L, h_KS)

  # for the first guess i'll just ignore the need for v_xc and v_h.
  # T+diag(v_ext) is the Kohn-Sham Hamiltonian
  T = T_1e(N_KS, h_KS)
  V = v_ext(Z, x_KS)
  min_eps, min_phi = one_electron(V, h_KS, N_KS)
  # density is n
  n = 2*min_phi*np.conj(min_phi)

  # The self-consistent loop.
  # v_H uses the density of both electrons, so it also counts each electron repelling itself.
  # Both electrons share one orbital, so exactly half of v_H is self-repulsion: v_x = -v_H/2, v_c = 0.
  eps = 0 # eigenvalue of kohn sham, NOT actual energy
  errs = [] # convergence history, for the mixing parameter study

  for it in range(max_iter): # capped, so a diverging run (e.g. alpha too large) cannot loop forever
    v_H = V_H(n,h_KS,x_KS)
    v_XC = -0.5*v_H

    # Now utilising the v_H and v_XC
    min_new_eps, new_phi = s.linalg.eigh_tridiagonal(T.diagonal()+V+v_H+v_XC, T.diagonal(k=1), select="i", select_range=(0,0))
    new_phi = new_phi[:,0]/np.sqrt(h_KS)

    # density is n
    n_new = 2*new_phi*np.conj(new_phi)

    errs.append(np.linalg.norm(n_new - n))
    if np.linalg.norm(n_new - n) < err_thres:
      n = n_new
      eps = min_new_eps[0]
      break
    else:
      n = alpha*n_new + (1-alpha)*n
  converged = errs[-1] < err_thres

  # Energy from converged n
  v_H = V_H(n, h_KS, x_KS) # Hartree potential
  E_H = 0.5 * h_KS * (n@v_H) # Hartree energy, avoiding double counting
  E_x = -0.5*E_H # same reasoning as for v_x earlier
  E_c = 0.0
  T_s = 2*h_KS* (np.conj(new_phi) @ (T @ new_phi)) # "kinetic energy" of non interacting electrons
  E_ext = h_KS * (n@V) # due to nuclear attraction

  E_KS = T_s + E_ext + E_H + E_x + E_c

  return {"E": E_KS, "E_check": 2*eps + E_x, "T_s": T_s, "E_ext": E_ext, "E_H": E_H, "E_x": E_x, "E_c": E_c,
          "eps": eps, "n": n, "phi": new_phi, "x": x_KS, "h": h_KS, "v_H": v_H, "v_XC": -0.5*v_H,
          "iters": len(errs), "errs": errs, "converged": converged}


if __name__ == "__main__":
  res = run(Z=1)
  print("integral of n        :", res["h"] * np.sum(res["n"]))      # must be 2
  print("E_KS (term by term)  :", res["E"])
  # To see if any unexpected behaviour is there.
  print("E_KS (2*eps - E_H/2) :", res["E_check"])
  print("iterations           :", res["iters"])
  plt.plot(res["x"], res["n"])
  plt.xlabel("x (bohr)")
  plt.ylabel("n(x)")
  plt.title("Kohn-Sham (= RHF) density, H-")
  plt.show()
