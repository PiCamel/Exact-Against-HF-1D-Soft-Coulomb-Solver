# Kohn-Sham (= restricted Hartree-Fock) process for an H2 molecule. Two nuclei of charge 1 at x = +-R/2.
# What changes from KS_H_minus.py: v_ext now has two wells, and the total energy gets an extra
# nucleus-nucleus repulsion w(R). For 2 electrons in one orbital, v_x = -v_H/2 still holds exactly.

import numpy as np
import scipy as s
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) # so common.py (one folder up) can be imported
from common import make_grid, w_KS, v_ext_H2, T_1e, V_H, one_electron


def run(R=3.5, L=10.0, h=0.1, err_thres=1e-6, max_iter=1000, symmetrize=True):
  alpha = 0.4 # conservative convex combination for the density mixing
  R_H2 = R # internuclear distance (nuclei at +-R/2)
  L_H2 = L # use L = R/2 + 9 for large R, the box must grow with R
  h_H2 = h
  x_H2, N_H2 = make_grid(L_H2, h_H2)

  # for the first guess i'll again ignore v_H and v_xc.
  T_H2 = T_1e(N_H2, h_H2)
  V_H2 = v_ext_H2(x_H2, R_H2)
  eps_H2, phi_H2 = one_electron(V_H2, h_H2, N_H2)
  # density is n
  n_H2 = 2*phi_H2*np.conj(phi_H2)

  # Self Consistent KS Loop
  eps_H2 = 0 # eigenvalue of kohn sham, NOT actual energy
  errs = [] # convergence history

  for it in range(max_iter): # capped, so a run that never settles (charge sloshing) cannot loop forever
    v_H_H2 = V_H(n_H2, h_H2, x_H2)
    v_XC_H2 = -0.5*v_H_H2

    # Now utilising the v_H and v_XC
    min_new_eps_H2, new_phi_H2 = s.linalg.eigh_tridiagonal(T_H2.diagonal()+V_H2+v_H_H2+v_XC_H2, T_H2.diagonal(k=1), select="i", select_range=(0,0))
    new_phi_H2 = new_phi_H2[:,0]/np.sqrt(h_H2)

    # density is n
    n_new_H2 = 2*new_phi_H2*np.conj(new_phi_H2)
    # the molecule is mirror symmetric, so the true density is too. Forcing it stops the
    # charge from "sloshing" between the two nuclei every iteration at large R.
    # (symmetrize=False is kept only to demonstrate the sloshing in compare.py)
    if symmetrize:
      n_new_H2 = 0.5*(n_new_H2 + n_new_H2[::-1])

    errs.append(np.linalg.norm(n_new_H2 - n_H2))
    if np.linalg.norm(n_new_H2 - n_H2) < err_thres:
      n_H2 = n_new_H2
      eps_H2 = min_new_eps_H2[0]
      break
    else:
      n_H2 = alpha*n_new_H2 + (1-alpha)*n_H2
  converged = errs[-1] < err_thres

  # Energy from converged n
  v_H_H2 = V_H(n_H2, h_H2, x_H2) # Hartree potential
  E_H_H2 = 0.5 * h_H2 * (n_H2@v_H_H2) # Hartree energy, avoiding double counting
  E_x_H2 = -0.5*E_H_H2 # same reasoning as for v_x earlier
  E_c_H2 = 0.0
  T_s_H2 = 2*h_H2* (np.conj(new_phi_H2) @ (T_H2 @ new_phi_H2)) # "kinetic energy" of non interacting electrons
  E_ext_H2 = h_H2 * (n_H2@V_H2) # due to attraction to both nuclei
  E_nn_H2 = w_KS(R_H2) # repulsion between the two nuclei

  E_KS_H2 = T_s_H2 + E_ext_H2 + E_H_H2 + E_x_H2 + E_c_H2 + E_nn_H2

  return {"E": E_KS_H2, "E_check": 2*eps_H2 + E_x_H2 + E_nn_H2, "T_s": T_s_H2, "E_ext": E_ext_H2,
          "E_H": E_H_H2, "E_x": E_x_H2, "E_c": E_c_H2, "E_nn": E_nn_H2, "eps": eps_H2,
          "n": n_H2, "phi": new_phi_H2, "x": x_H2, "h": h_H2, "R": R_H2, "iters": len(errs), "errs": errs, "converged": converged}


if __name__ == "__main__":
  res = run(R=3.5)
  print("integral of n        :", res["h"] * np.sum(res["n"]))      # must be 2
  print("E_KS (term by term)  :", res["E"])
  # To see if any unexpected behaviour is there.
  print("E_KS (2*eps + E_x + E_nn) :", res["E_check"])
  print("iterations           :", res["iters"])
  plt.plot(res["x"], res["n"])
  plt.xlabel("x (bohr)")
  plt.ylabel("n(x)")
  plt.title("Kohn-Sham (= RHF) density, H2 (R = 3.5)")
  plt.show()
