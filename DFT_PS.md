# Computational Physics Project: DFT and Electronic-Structure Problem Statements (D1-D4)

Project: group of 3, submission Nov 3. Required: (1) pick a problem needing numerics, (2) explain why computation is necessary, (3) implement a method, (4) present and analyze results, (5) at least one extension or parameter study, (6) presentation.

Effort figures are my rough estimates in person-hours for one working implementation plus one extension. Adjust to your group's skill.

| | Problem | Novelty | Risk | Effort (est.) |
|---|---|---|---|---|
| D1 | Kohn-Sham LDA for atoms, validated against NIST | medium | low | 25-30 h |
| D2 | 1D soft-Coulomb two-electron system: exact vs Kohn-Sham | high | medium | 30-40 h |
| D3 | Band structure by empirical pseudopotential (Si, Ge, GaAs) | low-medium | lowest | 15-20 h |
| D4 | Hartree-Fock and DFT for H2 and HeH+ in a Gaussian basis | low | low | 20-25 h |

---

## 0. Shared background (read first, about 1 hour)

**The core problem.** An N-electron system has a wavefunction Psi(r1,...,rN) in 3N dimensions. For Ne (N=10) that is 30 dimensions. A coarse 100-point grid per dimension would need 100^30 numbers, so exact solutions are impossible beyond tiny systems. This is the "why computation is necessary" argument, and it applies to every problem below.

**The DFT idea.**
- Hohenberg-Kohn theorem: the ground-state density n(r), a function of only 3 variables, determines everything.
- Kohn-Sham (KS) trick: replace the interacting electrons by fictitious non-interacting electrons that reproduce the same density. They move in

```
v_eff(r) = v_ext(r) + v_H[n](r) + v_xc[n](r)
```

  - `v_ext`: nuclear attraction.
  - `v_H`: classical electrostatic repulsion of the density cloud (Hartree).
  - `v_xc`: everything quantum left over (exchange and correlation). It is not known exactly. LDA uses the xc energy of a uniform electron gas evaluated at the local density.
- v_eff depends on n, and n depends on the orbitals, which depend on v_eff. So you iterate (the SCF loop):

```
guess n(r)
repeat:
    build v_H[n], v_xc[n]  ->  v_eff
    solve [ -1/2 d^2/dx^2 + v_eff ] phi_i = eps_i phi_i
    n_new = sum_i occ_i |phi_i|^2
    n = alpha * n_new + (1 - alpha) * n        # mixing, alpha ~ 0.3-0.5
until |n_new - n| < tol
```

**Units.** Hartree atomic units (hbar = m = e = 1). Energy in Hartree (1 Ha = 27.211 eV), length in bohr (1 bohr = 0.529 A). Convert only for final plots.

**General resources**
- Capelle, *A Bird's-Eye View of Density-Functional Theory*, arXiv:cond-mat/0211443. Best first read; free.
- Burke, *The ABC of DFT* (free PDF on his UC Irvine page). More formal.
- Burke, "Perspective on density functional theory", J. Chem. Phys. 136, 150901 (2012).
- Kohn, Nobel lecture (1998).
- Thijssen, *Computational Physics* (Schrodinger solvers and DFT chapters).

---

## D1. Self-consistent Kohn-Sham LDA for atoms, validated against NIST

**What you build.** A roughly 150-line Python program that computes total energies and electron densities of He, Be, Ne, Mg, Ar, etc., then checks them against published numbers.

**Why it is tractable.** For a spherical atom, write each orbital as psi = (u(r)/r) Y_lm. Then the 3D problem becomes a 1D radial eigenproblem:

```
-1/2 u'' + [ l(l+1)/(2 r^2) + v_eff(r) ] u = eps u,     u(0) = u(inf) = 0
```

**Ingredients**
1. **Grid and eigensolver.** Start with a uniform grid (r from about 1e-3 to 20-30 bohr, 5000-10000 points) and `scipy.linalg.eigh_tridiagonal`. The 1/r singularity makes convergence in the grid spacing slow near the nucleus, so run a convergence study. The upgrade is a logarithmic grid r = r0 e^x, which gives much better accuracy but turns the problem into a generalized eigenproblem. Do this only if time allows.
2. **Hartree potential.** No ODE solver is needed:
   `v_H(r) = (1/r) * integral_0^r n(r') 4 pi r'^2 dr' + integral_r^inf n(r') 4 pi r' dr'`
   Use `scipy.integrate.cumulative_trapezoid`.
3. **LDA exchange (Dirac).** `e_x = -(3/4)(3/pi)^(1/3) n^(1/3)`, `v_x = -(3 n / pi)^(1/3)`.
4. **LDA correlation.** Perdew-Zunger 1981 or VWN, closed-form functions of `r_s = (3/(4 pi n))^(1/3)`. Look up the coefficients and cross-check against libxc or PySCF. Coefficient typos are the number-one time sink.
5. **Total energy.** `E = sum_i eps_i - 1/2 * integral n v_H - integral n v_xc + E_xc[n]`. The classic bug is double-counting the Hartree term.
6. **Occupations.** Start with closed-shell atoms (He, Be, Ne, Mg, Ar). Open-shell atoms need spherically averaged fractional occupations, which makes a good extension.

**Validation targets (spin-unpolarized LDA, Ha).** He -2.8348, Ne -128.233. Hydrogen is a trap: the NIST value -0.4787 Ha is *spin-polarized* LSDA. Spin-unpolarized LDA gives about -0.4457 Ha. Compare like with like. Match the correlation functional to the one the reference table uses.

**Extensions and parameter studies**
- Scan Z = 1-18: plot r^2 n(r) to show shell structure; plot ionization energies against experiment and discuss the errors.
- Show the LDA potential decays exponentially instead of -1/r (a standard limitation).
- Self-interaction error: for H, E_LDA differs from the exact -0.5 Ha although there is only one electron.
- SCF convergence versus mixing parameter alpha.
- Compare Thomas-Fermi, exchange-only, and full LDA.
- Cross-check against PySCF (LDA and PBE). Finite Gaussian bases will not match a numerical radial solver to micro-Hartree; use a large basis and expect mHa-level agreement.

**Resources**
- NIST "Atomic Reference Data for Electronic Structure Calculations" (physics.nist.gov/PhysRefData/DFTdata); verify the exact URL when you open it.
- PySCF docs (pyscf.org): `dft.RKS`.
- Thijssen, atom chapter.

---

## D2. 1D soft-Coulomb two-electron system: exact vs Kohn-Sham

**What you build.** An exactly solved two-electron system on a grid, then the same system treated with KS-DFT, so you can measure where approximate DFT fails against a true reference. Real 3D atoms never allow this beyond He.

**The system.** Two electrons on a line, nucleus of charge Z at x = 0, all Coulomb interactions softened with `w(x) = 1/sqrt(x^2 + 1)`:

```
H = -1/2 d^2/dx1^2 - 1/2 d^2/dx2^2 - Z w(x1) - Z w(x2) + w(x1 - x2)
```

- The Hamiltonian has no spin operators. Its ground state is nodeless, hence symmetric in x1 <-> x2, hence paired with the spin singlet. You solve for one real, positive, symmetric Psi(x1, x2).
- Density: `n(x) = 2 * integral |Psi(x, x')|^2 dx'`, with integral n = 2.

**Why computation is necessary (honest framing).** Even the one-electron soft-Coulomb problem has no closed form. The two-electron problem is a 2D PDE with no analytic solution. A grid solves it easily (N_x^2 unknowns), which is exactly why it makes a good benchmark. For N electrons the cost is N_x^N, which is why DFT exists. Do not claim two electrons cannot be solved; claim that a computable exact answer lets you test approximations quantitatively.

### Building blocks

**M1. One-electron solver.** Uniform grid on [-L, L], 3-point finite-difference kinetic energy, `eigh_tridiagonal`. Use h = 0.1-0.2, L about 10 for atoms. Converges as O(h^2). Extrapolate with `E ~ E_h + (E_h - E_2h)/3`.

**M2. Exact two-electron solver.**
```python
T  = sp.diags([-0.5/h**2*np.ones(N-1), 1/h**2*np.ones(N), -0.5/h**2*np.ones(N-1)], [-1, 0, 1])
I  = sp.identity(N)
X1, X2 = np.meshgrid(x, x, indexing='ij')
V  = np.tile(v, N) + np.repeat(v, N) + w(X1 - X2).ravel()
H  = sp.kron(T, I) + sp.kron(I, T) + sp.diags(V)
E, P = scipy.sparse.linalg.eigsh(H.tocsc(), k=1, which='SA', tol=1e-10)
Psi = P[:, 0].reshape(N, N) / h
n   = 2 * (Psi**2).sum(axis=1) * h
```
N = 201 gives 40k unknowns and runs in seconds. Fix the sign of Psi and check that the density integrates to 2. Free extension: the lowest *antisymmetric* eigenstate is the triplet, so the singlet-triplet gap costs nothing extra.

**M3. KS with exact exchange (equals restricted Hartree-Fock for two electrons).** With one doubly occupied orbital phi and n = 2 phi^2:
- `v_x = -v_H / 2`, so `v_eff = v_ext + v_H / 2`.
- `v_H(x) = integral n(x') w(x - x') dx'` (a matrix-vector product).
- `E = 2<phi|T|phi> + integral n v_ext + E_H + E_x`, with `E_H = 1/2 integral n v_H` and `E_x = -E_H / 2`.
- The *exact correlation energy* is `E_c = E_exact - E_HF`, with no QMC needed.

**M4. LDA exchange for the 1D soft-Coulomb gas (derive it yourselves).** Uniform unpolarized gas, `k_F = pi n / 2`, density matrix `gamma_sigma(x) = (k_F/pi) sin(k_F x)/(k_F x)`. The exchange energy density is

```
n e_x(n) = -(n^2 / 2) J(k_F),     J(k) = integral_0^inf  sin^2(t) / ( t^2 sqrt(t^2 + k^2) ) dt
```

Tabulate on a log-spaced n grid, build a CubicSpline in log n, and take `v_x = d(n e_x)/dn` from the spline derivative. Check: e_x -> -1/2 as n -> infinity.

**M5. LDA correlation.** The parametrization for exactly this interaction is in Helbig, Fuks, Casula, Verstraete, Marques, Tokatly, Rubio, Phys. Rev. A 83, 032503 (2011), arXiv:1101.2564 (QMC-based fit, a = 1, unpolarized and polarized). Transcribe the coefficients from Table I of the paper. In 1D, `r_s = 1/(2n)`. Validate by reproducing the 1D He energy they report. Fallback if this stalls: drop LDA-C; the project still works with LDA-X against exact, since the exact E_c comes from M3.

**M6. Exact xc potential (no iteration needed for N = 2).** For a singlet pair, `phi = sqrt(n/2)` exactly, so the KS equation can be solved for the potential directly:

```
v_s(x) = eps + phi''(x) / (2 phi(x)),      v_xc = v_s - v_ext - v_H,      v_c = v_xc + v_H / 2
```

- Fix the constant `eps` so v_xc -> 0 at large |x|. In exact DFT, `eps = -I`.
- Restrict to the region n > about 1e-8, because phi''/phi amplifies noise in the tails.
- Plot exact v_xc against the LDA v_x. The LDA potential decays exponentially; the exact one decays as -1/|x|.
- Optional method comparison: Zhao-Morrison-Parr iteration `v_s <- v_s + alpha (n_KS - n_exact)` (slow and noisy), or Wu-Yang. Use it only to cross-check the direct formula.

### Reference numbers (from my own calculation, h = 0.1 with Richardson extrapolation; reproduce before trusting)

| Quantity | Value (Ha) |
|---|---|
| 1D H atom (Z = 1, one electron) | -0.6698 |
| 1D He+ (Z = 2, one electron) | -1.4834 |
| 1D He, exact | -2.2383 |
| 1D He, RHF (= KS-EXX) | -2.2242 |
| Exact correlation energy E_c | -0.0140 |
| Exact ionization energy I | 0.7548 |
| 1D He, LDA exchange-only | -2.158 |
| E_x: LDA vs exact (-E_H/2) | -0.654 vs -0.724 |

LDA-X underestimates exchange by about 10%, while correlation is only about 0.014 Ha. So in 1D He most of the LDA error is exchange. I recall the H and He values matching Helbig et al. and Wagner et al. (about -0.670 and -2.238), but check the paper tables.

### Extension: stretched H2 (strongest extension)

Two soft nuclei at +/- R/2 (Z = 1 each) with nuclear repulsion `w(R)` added. Grow the box with R (L = R/2 + 9 worked). My results (h = 0.15, accuracy a few 1e-4 Ha):

| R | Exact | RHF (= KS-EXX) | LDA-X only |
|---|---|---|---|
| 1 | -1.4134 | -1.3947 | -1.3331 |
| 2 | -1.4421 | -1.4060 | -1.3561 |
| 4 | -1.3509 | -1.1904 | -1.1281 |
| 6 | -1.3395 | -1.0693 | -1.0729 |
| 10 | -1.3398 | -1.0224 | -1.0651 |

- The exact curve has a minimum near R = 2 and dissociates to 2 x (-0.6698) = -1.3396.
- RHF dissociates about 0.32 Ha too high. The exact E_c grows from -0.015 at R = 0.5 to -0.317 at R = 10. This is static correlation: the restricted wavefunction forces 50% ionic character at any R.
- LDA-X also fails by about 0.27 Ha at R = 10. I have not run full LDA-XC; expect the same failure, and treat it as a test of the claim that a better xc functional does not fix static correlation error.
- Plot exact v_xc(x) at R = 1, 4, 8. At large R a peak/step develops at the bond midpoint, which no LDA reproduces. This is the highlight figure.

### Parameter studies (pick two or three)
1. Grid convergence in h and L (mandatory; it gives your error bar).
2. SCF mixing alpha versus iteration count, including divergence near alpha = 1.
3. Z-scan for two-electron ions: H-, He, Li+, Be2+. Plot LDA-X relative error and |E_c| against Z (high-density limit).
4. Softening parameter a beyond a = 1.
5. HOMO test: exact -eps_HOMO equals I = 0.7548. LDA should give a smaller value because of the wrong asymptotic decay.

### Pitfalls
- Wrong sign or scale on v_x: check E_x(LDA) against -E_H/2.
- Energy double counting: use `E = T_s + integral n v_ext + E_H + E_xc`, not a sum of eigenvalues alone.
- Box too small at large R: if the energy drifts as L grows, enlarge it.
- Memory: N_x^2 grows fast; R = 10 is about 70k unknowns, still fine.
- Scope: finish M1-M3 first. That alone is a complete, defensible project before touching M4-M6.

### Resources
- Capelle, arXiv:cond-mat/0211443.
- Helbig et al., PRA 83, 032503 (2011), arXiv:1101.2564 (LDA parametrization, 1D He benchmark).
- Wagner, Stoudenmire, Burke, White, Phys. Chem. Chem. Phys. 14, 8581 (2012), "Reference electronic structure calculations in one dimension". Check which interaction they use before copying numbers; it may differ from `1/sqrt(x^2+1)`.
- Cohen, Mori-Sanchez, Yang, Science 321, 792 (2008): delocalization and static correlation error.
- Burke, J. Chem. Phys. 136, 150901 (2012).
- Wu and Yang, J. Chem. Phys. 118, 2498 (2003): potential inversion.

---

## D3. Band structure by empirical pseudopotential (Si, Ge, GaAs)

**What you build.** The band structure of silicon (why it is a semiconductor with a gap of about 1.1 eV, indirect) by diagonalizing a matrix at each k-point in a plane-wave basis. This is not DFT, but it is the same plane-wave machinery used by Quantum ESPRESSO and VASP.

**Physics.** For a periodic potential `V(r) = sum_G V(G) e^{iG.r}`, Bloch's theorem gives `psi_k(r) = sum_G c_G e^{i(k+G).r}`. For each k, solve

```
sum_G' [ (|k+G|^2 / 2) delta_{G,G'} + V(G - G') ] c_G' = E c_G
```

**Why computation.** A periodic potential with a two-atom basis has no closed-form band structure. Nearly-free-electron theory works only perturbatively and fails to give quantitative gaps.

**Ingredients**
1. FCC lattice (a = 5.43 A for Si); the reciprocal lattice is BCC. Generate all G vectors up to a cutoff (about 100-250 plane waves), in units of 2 pi / a.
2. Potential matrix from a few form factors: `V(G - G') = V_S(|dG|^2) cos(dG.tau) + i V_A(|dG|^2) sin(dG.tau)`, `tau = (a/8)(1,1,1)`. For Si only `|dG|^2 = 3, 8, 11` contribute and V_A = 0. Cohen-Bergstresser Si values are about V3 = -0.21, V8 = +0.04, V11 = +0.08 Ry. Check these in the paper; units are Rydberg.
3. Loop k along Gamma-X-W-L-Gamma-K, diagonalize with `numpy.linalg.eigh`, plot the lowest 8 bands. Convert Ry to Ha or eV carefully.

**Validation.** Si: valence-band maximum at Gamma, conduction-band minimum about 85% of the way along Gamma-X (indirect), gap on the order of 1 eV. GaAs: direct gap at Gamma; needs the antisymmetric form factors.

**Extensions**
- Convergence versus plane-wave cutoff.
- Gap versus lattice constant (pressure dependence, deformation potentials).
- Effective masses from band curvature (finite differences).
- Density of states by Gaussian smearing or tetrahedron method.
- Si vs Ge vs GaAs comparison.
- Optional bridge to real DFT: compare with a GPAW or Quantum ESPRESSO band structure.

**Assessment.** Lowest risk and runs in minutes on a laptop. Weakness: it is a standard textbook exercise, so originality has to come from the extensions.

**Resources**
- Cohen and Bergstresser, Phys. Rev. 141, 789 (1966).
- Chelikowsky and Cohen, Phys. Rev. B 14, 556 (1976).
- Yu and Cardona, *Fundamentals of Semiconductors* (pseudopotential chapter).
- Kittel, *Introduction to Solid State Physics* (Bloch theorem, nearly-free-electron model).

---

## D4. Hartree-Fock and DFT for H2 and HeH+ in a Gaussian basis

**What you build.** The textbook Roothaan SCF procedure with a minimal Gaussian basis, a dissociation curve, and a demonstration of restricted Hartree-Fock's failure for H2. The extension adds an LDA exchange-correlation term on a numerical grid as a bridge to DFT.

**Method.**
- Expand orbitals in STO-3G (each Slater function as 3 Gaussians). All integrals for s-type Gaussians have closed forms using the Boys function `F0(t) = (1/2) sqrt(pi/t) erf(sqrt(t))`.
- Solve the generalized eigenproblem `F C = S C eps` and iterate to self-consistency.
- Szabo and Ostlund chapter 3 and Appendix A give the integral formulas.

**Validation.** H2 at R = 1.4 bohr, STO-3G: E about -1.1167 Ha. HeH+ in STO-3G near R = 1.46 bohr gives about -2.86 Ha; check both against Szabo and Ostlund and against PySCF (which should agree to about 1e-6).

**Extensions**
- Plot E(R). RHF H2 dissociates to the wrong limit (several tenths of a Ha too high, from spurious ionic terms). Note that RHF HeH+ dissociates correctly to He + H+, which is an instructive contrast.
- Add an LDA xc term via numerical integration on a molecular grid (the DFT bridge); compare to the RHF curve.
- Optional: unrestricted HF, which fixes the dissociation limit at the cost of spin contamination.

**Assessment.** Classic, well documented, low risk, but it is a rederivation of textbook material. The two-electron-integral bookkeeping is where beginners get stuck.

**Resources**
- Szabo and Ostlund, *Modern Quantum Chemistry*, ch. 3 and Appendix A.
- Crawford Group Programming Projects (github.com/CrawfordGroup/ProgrammingProjects).
- PySCF docs (pyscf.org) for validation.

---

## Recommendation

- **D2 as the main project**, because the exact reference makes the error analysis quantitative and the stretched-H2 extension is a strong "limitations of the method" argument.
- **D1 as the lower-risk alternative** (published targets at every step), or as a validation/extension block if time allows.
- D3 is the safety net if the group dislikes SCF. D4 is the most textbook-like.
- Suggested split for 3 people: one on the solver(s), one on the KS/SCF side, one on reference runs, plotting and the slide skeleton from day 1.