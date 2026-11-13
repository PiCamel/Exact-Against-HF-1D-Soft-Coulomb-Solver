"""
LDA_He.py  --  TO DO: Kohn-Sham with LDA exchange for the He atom (one nucleus, Z = 2)

WHAT THIS FILE SHOULD DO
------------------------
Exactly what LDA_H_minus.py does, with Z = 2 instead of Z = 1. Do LDA_H_minus.py first: every step
(the J(k) table, the spline, v_x, the loop and the energy) is written out there.
Easiest start: copy your finished LDA_H_minus.py into this file, set Z = 2 inside run(), and drop Z from
the arguments, the same way KS_He.py differs from KS_H_minus.py.

If you put the LDA table helpers (nex_LDA, v_x_LDA) in common.py, import them from there. If they live in
LDA_H_minus.py, import them with:
    from LDA.LDA_H_minus import nex_LDA, v_x_LDA
(this works because the path line at the top of the file puts the project folder on sys.path).
Keep the path line from KS_He.py so common.py can be found:
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


WHAT run() MUST RETURN, so compare.py can use it
------------------------------------------------
  def run(L=10.0, h=0.1, alpha=0.4, err_thres=1e-6, max_iter=1000, symmetrize=True):
      ...
      return {"E", "E_check", "T_s", "E_ext", "E_H", "E_x", "E_c", "eps", "n", "phi", "x", "h",
              "v_H", "v_XC", "iters", "errs", "converged"}    (as a dict with these keys)
  plus an  if __name__ == "__main__":  block that prints the energies and plots n(x), like KS_He.py.


TARGET NUMBERS (h = 0.1, L = 10, alpha = 0.4)
---------------------------------------------
  E   = -2.15845
  E_x = -0.6535   (vs -E_H/2 = -0.7190 with exact exchange: LDA misses about 9% of the exchange)
  eps = -0.4673   (vs the exact ionization energy I = 0.7549: the LDA HOMO is far too shallow)
  about 25 iterations
  compare: KS exact exchange -2.22450, exact -2.23855

Unlike H-, LDA does bind He (eps < 0), so there is no box-size problem here.
"""
