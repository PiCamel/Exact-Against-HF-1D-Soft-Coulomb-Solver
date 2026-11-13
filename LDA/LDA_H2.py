"""
LDA_H2.py  --  TO DO: Kohn-Sham with LDA exchange for the H2 molecule (nuclei of charge 1 at x = -R/2 and +R/2)

WHAT THIS FILE SHOULD DO
------------------------
Same as KS_H2.py, with LDA exchange instead of exact exchange. Do LDA_H_minus.py first: this file reuses the
same LDA table and the same changes to the loop and the energy. Easiest start: copy KS_H2.py into this file.


STEP 1 -- the LDA pieces
------------------------
Use the same n e_x(n) table and v_x(n) as in LDA_H_minus.py. If you wrote them in common.py, import them;
if you wrote them in LDA_H_minus.py, import them from there (from LDA.LDA_H_minus import nex_LDA, v_x_LDA).
Build the table once; never inside the loop.


STEP 2 -- the self-consistent loop (changes from KS_H2.py)
-----------------------------------------------------------
  - v_XC_H2 = v_x_LDA(n_H2)      instead of  -0.5*v_H_H2
  - FULL v_H in the Hamiltonian: T_H2.diagonal() + V_H2 + v_H_H2 + v_XC_H2
  - keep the density symmetrization n_new_H2 = 0.5*(n_new_H2 + n_new_H2[::-1])
    (without it the charge sloshes from one nucleus to the other at large R and never converges)
  - for it in range(max_iter) instead of while True; record converged


STEP 3 -- the energy
--------------------
  E_H_H2  = 0.5 * h * (n_H2 @ v_H_H2)
  E_x_H2  = h * np.sum(nex_LDA(n_H2))        NEW
  T_s_H2, E_ext_H2                           same as KS_H2.py
  E_nn_H2 = w_KS(R)                          nucleus-nucleus repulsion, a constant at fixed nuclei
  E       = T_s_H2 + E_ext_H2 + E_H_H2 + E_x_H2 + E_nn_H2

  Check (general Kohn-Sham eigenvalue route):
      E_check = 2*eps_H2 - E_H_H2 - h*(n_H2 @ v_XC_H2) + E_x_H2 + E_nn_H2
  It must agree with E.


STEP 4 -- what run() must return, so compare.py can use it
----------------------------------------------------------
  def run(R=3.5, L=10.0, h=0.1, alpha=0.4, err_thres=1e-6, max_iter=1000):
      ...
      return {"E", "E_check", "T_s", "E_ext", "E_H", "E_x", "E_c", "E_nn", "eps", "n", "phi", "x", "h", "R",
              "iters", "errs", "converged"}    (as a dict with these keys)
  plus an  if __name__ == "__main__":  block that prints the energies and plots n(x) for R = 3.5.

  The box must grow with R: use L = max(10, R/2 + 9).


TARGET NUMBERS (h = 0.1, alpha = 0.4, total energy including E_nn, in Ha)
-------------------------------------------------------------------------
    R      L      exact      KS exact exchange    KS-LDA
   1.0   10.0   -1.41320        -1.39452          -1.33295
   2.0   10.0   -1.44200        -1.40591          -1.35599
   3.5   10.0   -1.36357        -1.26423          -1.23813
   6.0   12.0   -1.33930        -1.12054          -1.13913
  10.0   14.0   -1.33963        -1.05706          -1.12056
  Exact limit R -> infinity: 2 x E(H atom) = -1.3397.
  At R = 3.5 the LDA HOMO is eps = -0.3512 and the loop takes about 23 iterations.


WHAT TO LOOK FOR
----------------
  - LDA sits above exact exchange near equilibrium (exchange is underestimated: self-interaction error),
    but below it at large R. Both mean-field curves end far above the exact limit: static correlation,
    which no local functional fixes.
  - LDA's density at R = 3.5 lies between the exact and the KS one in the bond region (plot all three).
  - -eps (LDA) is much smaller than the true ionization energy: the LDA potential dies off too fast.
"""
