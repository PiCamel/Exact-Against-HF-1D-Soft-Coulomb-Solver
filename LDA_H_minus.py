"""
LDA_H_minus.py  --  TO DO: Kohn-Sham with LDA exchange for one nucleus (Z = 1 is H-, Z = 2 is He)

WHAT THIS FILE SHOULD DO
------------------------
Same self-consistent loop as KS_H_minus.py. The ONLY physics change is the exchange part:
    KS_H_minus.py : v_x = -v_H/2                    (exact exchange; exact only because both electrons share one orbital)
    this file     : v_x(x) = d[n e_x(n)]/dn at n = n(x)   (LDA: each point treated as a piece of uniform electron gas)
Correlation stays out (v_c = 0, E_c = 0), so this is "LDA exchange only" (LDA-X).
Easiest start: copy KS_H_minus.py into this file and change the parts listed below.


STEP 1 -- the uniform-gas exchange energy (the "LDA table")
-----------------------------------------------------------
For a uniform, unpolarised 1D gas with the soft-Coulomb interaction w(x) = 1/sqrt(x^2 + 1):
    k_F = pi n / 2
    n e_x(n) = -(n^2 / 2) * J(k_F)
    J(k)     = integral from 0 to infinity of  sin^2(t) / ( t^2 * sqrt(t^2 + k^2) )  dt
(Derive this for the report: put the uniform-gas density matrix gamma(s) = sin(k_F s)/(pi s) per spin into
 E_x = -(1/2) sum_spins integral w(s) |gamma(s)|^2 ds, then substitute t = k_F s.)

J(k) has no closed form, so compute it numerically:
  - write the integrand with np.sinc(t/np.pi)**2 / np.sqrt(t*t + k*k)   (np.sinc(t/pi) = sin(t)/t, safe at t = 0)
  - integrate with scipy.integrate.quad one oscillation at a time: intervals [0, pi], [pi, 2 pi], ... up to T = 200
  - add the tail beyond T analytically: about 1/(4 T^2)

Then build the table ONCE (it is slow, so do not rebuild it every iteration):
  - n_tab = np.logspace(-10, log10(30), 300)
  - f_tab = n e_x(n) for every n in n_tab
  - spline f against ln n:  spl = CubicSpline(np.log(n_tab), f_tab)
  - n e_x(n) at any density:  spl(np.log(n))
  - v_x(n) = d f / d n = (1/n) * d f / d ln n = spl(np.log(n), 1) / n
  - clip n into [n_tab[0], n_tab[-1]] before using the spline, and set both to 0 where n < n_tab[0] (the far tails)

Where to put them: write J, the table, nex_LDA(n) and v_x_LDA(n) either at the top of this file or in
common.py (then LDA_H2.py can import them too). Imports you will need: from scipy.integrate import quad,
from scipy.interpolate import CubicSpline. Tip: build the table in a function and cache it
(functools.lru_cache) or as a module-level variable, so both files reuse it.

Checks before going on:
  - e_x = f_tab[-1] / n_tab[-1] should be about -0.497 at n = 30 (it tends to -1/2 as n -> infinity)
  - v_x from the spline at n = 0.5 should match a finite difference of n e_x(n): about -0.4485


STEP 2 -- the self-consistent loop (changes from KS_H_minus.py)
----------------------------------------------------------------
  - v_XC = v_x_LDA(n)           instead of  v_XC = -0.5*v_H
  - use the FULL v_H: T.diagonal() + V + v_H + v_XC   (LDA does not cancel self-repulsion exactly, so no 1/2)
  - keep the mixing n = alpha*n_new + (1-alpha)*n
  - use  for it in range(max_iter)  instead of  while True,  and record converged = (last error < err_thres)
  - symmetrize the new density each iteration: n_new = 0.5*(n_new + n_new[::-1])
    (one nucleus at x = 0, so the true density is mirror symmetric; this matters for H-, see the warning below)


STEP 3 -- the energy
--------------------
  E_H  = 0.5 * h * (n @ v_H)                 same as before
  E_x  = h * np.sum(nex_LDA(n))              NEW: no longer -E_H/2
  T_s, E_ext                                 same as before
  E    = T_s + E_ext + E_H + E_x             (E_c = 0)

  The shortcut E = 2*eps - E_H/2 only works for exact exchange. The general Kohn-Sham check is
      E_check = 2*eps - E_H - h*(n @ v_XC) + E_x
  and it must agree with E to about the convergence tolerance.


STEP 4 -- what run() must return, so compare.py can use it
----------------------------------------------------------
  def run(Z=1, L=10.0, h=0.1, alpha=0.4, err_thres=1e-6, max_iter=1000, symmetrize=True):
      ...
      return {"E", "E_check", "T_s", "E_ext", "E_H", "E_x", "E_c", "eps", "n", "phi", "x", "h",
              "v_H", "v_XC", "iters", "errs", "converged"}    (as a dict with these keys)
  plus an  if __name__ == "__main__":  block that prints the energies and plots n(x), like KS_H_minus.py.


TARGET NUMBERS (h = 0.1, L = 10, alpha = 0.4)
---------------------------------------------
  He (Z = 2):  E = -2.15845   E_x = -0.6535  (vs -E_H/2 = -0.7190)   eps = -0.4673   about 25 iterations
               compare: KS exact exchange -2.22450, exact -2.23855, exact ionization energy I = 0.7549
  H- (Z = 1):  E = -0.64409   eps = +0.129   about 32 iterations


WARNING -- LDA does NOT bind H-
-------------------------------
For Z = 1 the HOMO eigenvalue comes out POSITIVE. A positive eps means the second electron is not bound:
only the walls of the box hold it in. Make the box bigger (L = 8, 10, 12, 15, 20) and eps and E keep
changing; at large L the loop struggles to converge (use alpha = 0.1). This is not a bug. It is LDA's
self-interaction error: each electron partly repels itself, which is enough to unbind a weakly bound anion.
Exact exchange (KS_H_minus.py) has no self-interaction and binds H- (eps about -0.054).
Show this as a result: a table or plot of eps against L for both methods.
"""
