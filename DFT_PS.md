# Computational Physics Project: DFT and Electronic-Structure Problem Statements (D1–D4)

Project: group of 3, submission Nov 3. Required: (1) pick a problem needing numerics, (2) explain why computation is necessary, (3) implement a method, (4) present and analyze results, (5) at least one extension or parameter study, (6) presentation.

Effort figures are my rough estimates in person-hours for one working implementation plus one extension. Adjust to your group's skill.

| | Problem | Novelty | Risk | Effort (est.) |
|---|---|---|---|---|
| D1 | Kohn–Sham LDA for atoms, validated against NIST | medium | low | 25–30 h |
| D2 | 1D soft-Coulomb two-electron system: exact vs Kohn–Sham | high | medium | 30–40 h |
| D3 | Band structure by empirical pseudopotential (Si, Ge, GaAs) | low–medium | lowest | 15–20 h |
| D4 | Hartree–Fock and DFT for H$_2$ and HeH$^+$ in a Gaussian basis | low | low | 20–25 h |

---

## 0. Shared background (read first, about 1 hour)

**The core problem.** An $N$-electron system has a wavefunction $\Psi(\mathbf r_1,\dots,\mathbf r_N)$ in $3N$ dimensions. For Ne ($N=10$) that is 30 dimensions. A coarse grid of 100 points per dimension would need $100^{30}$ numbers, so exact solutions are impossible beyond tiny systems. This is the "why computation is necessary" argument, and it applies to every problem below.

**The DFT idea.**

- Hohenberg–Kohn theorem: the ground-state density $n(\mathbf r)$, a function of only 3 variables, determines everything.
- Kohn–Sham (KS) trick: replace the interacting electrons by fictitious non-interacting electrons that reproduce the same density. They move in

$$
v_\text{eff}(\mathbf r) = v_\text{ext}(\mathbf r) + v_H[n](\mathbf r) + v_{xc}[n](\mathbf r)
$$

  - $v_\text{ext}$: nuclear attraction.
  - $v_H$: classical electrostatic repulsion of the density cloud (Hartree).
  - $v_{xc}$: everything quantum left over (exchange and correlation). It is not known exactly. LDA uses the xc energy of a uniform electron gas evaluated at the local density.
- $v_\text{eff}$ depends on $n$, and $n$ depends on the orbitals, which depend on $v_\text{eff}$. So you iterate (the SCF loop). The KS equations and density are

$$
\left[-\tfrac12\nabla^2 + v_\text{eff}(\mathbf r)\right]\phi_i(\mathbf r) = \varepsilon_i\,\phi_i(\mathbf r),
\qquad
n(\mathbf r) = \sum_i f_i\,|\phi_i(\mathbf r)|^2
$$

```
guess n(r)
repeat:
    build v_H[n], v_xc[n]  ->  v_eff
    solve the KS eigenproblem for phi_i, eps_i
    n_new = sum_i f_i |phi_i|^2
    n = alpha * n_new + (1 - alpha) * n        # linear mixing, alpha ~ 0.3-0.5
until |n_new - n| < tol
```

**Units.** Hartree atomic units ($\hbar = m_e = e = 1$). Energy in Hartree ($1\ \text{Ha} = 27.211$ eV), length in bohr ($1\ a_0 = 0.529$ Å). Convert only for final plots.

**General resources**

- Capelle, *A Bird's-Eye View of Density-Functional Theory*, arXiv:cond-mat/0211443. Best first read; free.
- Burke, *The ABC of DFT* (free PDF on his UC Irvine page). More formal.
- Burke, "Perspective on density functional theory", J. Chem. Phys. 136, 150901 (2012).
- Kohn, Nobel lecture (1998).
- Thijssen, *Computational Physics* (Schrödinger solvers and DFT chapters).

---

## D1. Self-consistent Kohn–Sham LDA for atoms, validated against NIST

**What you build.** A roughly 150-line Python program that computes total energies and electron densities of He, Be, Ne, Mg, Ar, etc., then checks them against published numbers.

**Why it is tractable.** For a spherical atom, write each orbital as $\psi_{nlm} = \dfrac{u_{nl}(r)}{r}Y_{lm}(\theta,\varphi)$. Then the 3D problem becomes a 1D radial eigenproblem:

$$
-\frac12 u'' + \left[\frac{l(l+1)}{2r^2} + v_\text{eff}(r)\right]u = \varepsilon\,u,
\qquad u(0) = u(\infty) = 0
$$

**Ingredients**

1. **Grid and eigensolver.** Start with a uniform grid ($r$ from about $10^{-3}$ to 20–30 bohr, 5000–10000 points) and `scipy.linalg.eigh_tridiagonal`. The $1/r$ singularity makes convergence in the grid spacing slow near the nucleus, so run a convergence study. The upgrade is a logarithmic grid $r = r_0 e^{x}$, which gives much better accuracy but turns the problem into a generalized eigenproblem. Do this only if time allows.
2. **Hartree potential.** No ODE solver is needed:
$$
v_H(r) = \frac1r\int_0^r n(r')\,4\pi r'^2\,dr' + \int_r^\infty n(r')\,4\pi r'\,dr'
$$
   Use `scipy.integrate.cumulative_trapezoid`.
3. **LDA exchange (Dirac).**
$$
\varepsilon_x(n) = -\frac34\left(\frac3\pi\right)^{1/3} n^{1/3},
\qquad
v_x(n) = -\left(\frac{3n}{\pi}\right)^{1/3}
$$
4. **LDA correlation.** Perdew–Zunger 1981 or VWN, closed-form functions of the Wigner–Seitz radius
$$
r_s = \left(\frac{3}{4\pi n}\right)^{1/3}
$$
   Look up the coefficients and cross-check against libxc or PySCF. Coefficient typos are the number-one time sink.
5. **Total energy.**
$$
E = \sum_i f_i\,\varepsilon_i - \frac12\int n\,v_H\,d^3r - \int n\,v_{xc}\,d^3r + E_{xc}[n]
$$
   The classic bug is double-counting the Hartree term.
6. **Occupations.** Start with closed-shell atoms (He, Be, Ne, Mg, Ar). Open-shell atoms need spherically averaged fractional occupations $f_i$, which makes a good extension.

**Validation targets (spin-unpolarized LDA).** He: $-2.8348$ Ha. Ne: $-128.233$ Ha. Hydrogen is a trap: the NIST value $-0.4787$ Ha is *spin-polarized* LSDA. Spin-unpolarized LDA gives about $-0.4457$ Ha. Compare like with like. Match the correlation functional to the one the reference table uses.

**Extensions and parameter studies**

- Scan $Z = 1$–$18$: plot $r^2 n(r)$ to show shell structure; plot ionization energies against experiment and discuss the errors.
- Show that the LDA potential decays exponentially instead of as $-1/r$ (a standard limitation).
- Self-interaction error: for H, $E_\text{LDA} \neq -0.5$ Ha although there is only one electron.
- SCF convergence versus mixing parameter $\alpha$.
- Compare Thomas–Fermi, exchange-only, and full LDA.
- Cross-check against PySCF (LDA and PBE). Finite Gaussian bases will not match a numerical radial solver to micro-Hartree; use a large basis and expect mHa-level agreement.

**Resources**

- NIST "Atomic Reference Data for Electronic Structure Calculations" (physics.nist.gov/PhysRefData/DFTdata); verify the exact URL when you open it.
- PySCF docs (pyscf.org): `dft.RKS`.
- Thijssen, atom chapter.

---

## D2. 1D soft-Coulomb two-electron system: exact vs Kohn–Sham

**What you build.** An exactly solved two-electron system on a grid, then the same system treated with KS-DFT, so you can measure where approximate DFT fails against a true reference. Real 3D atoms never allow this beyond He.

**The system.** Two electrons on a line, nucleus of charge $Z$ at $x=0$, all Coulomb interactions softened with

$$
w(x) = \frac{1}{\sqrt{x^2+1}}
$$

$$
\hat H = -\frac12\frac{\partial^2}{\partial x_1^2} - \frac12\frac{\partial^2}{\partial x_2^2} - Z\,w(x_1) - Z\,w(x_2) + w(x_1 - x_2)
$$

- The Hamiltonian has no spin operators. Its ground state is nodeless, hence symmetric under $x_1 \leftrightarrow x_2$, hence paired with the spin singlet. You solve for one real, positive, symmetric $\Psi(x_1,x_2)$.
- Density:
$$
n(x) = 2\int |\Psi(x,x')|^2\,dx',\qquad \int n\,dx = 2
$$

**Why computation is necessary (honest framing).** Even the one-electron soft-Coulomb problem has no closed form. The two-electron problem is a 2D PDE with no analytic solution. A grid solves it easily ($N_x^2$ unknowns), which is exactly why it makes a good benchmark. For $N$ electrons the cost is $N_x^N$, which is why DFT exists. Do not claim two electrons cannot be solved; claim that a computable exact answer lets you test approximations quantitatively.

### Building blocks

**M1. One-electron solver.** Uniform grid on $[-L, L]$, 3-point finite-difference kinetic energy, `eigh_tridiagonal`. Use $h = 0.1$–$0.2$, $L \approx 10$ for atoms. Converges as $O(h^2)$. Extrapolate with

$$
E \approx E_h + \frac{E_h - E_{2h}}{3}
$$

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

$N = 201$ gives 40k unknowns and runs in seconds. Fix the sign of $\Psi$ and check that the density integrates to 2. Free extension: the lowest *antisymmetric* eigenstate is the triplet, so the singlet–triplet gap costs nothing extra.

**M3. KS with exact exchange (equals restricted Hartree–Fock for two electrons).** With one doubly occupied orbital $\varphi$ and $n = 2\varphi^2$:

$$
v_x = -\frac{v_H}{2},\qquad v_\text{eff} = v_\text{ext} + \frac{v_H}{2},\qquad v_H(x) = \int n(x')\,w(x-x')\,dx'
$$

$$
E = 2\langle\varphi|\hat T|\varphi\rangle + \int n\,v_\text{ext}\,dx + E_H + E_x,
\qquad E_H = \frac12\int n\,v_H\,dx,\quad E_x = -\frac{E_H}{2}
$$

$v_H$ is a matrix–vector product with the matrix $w(x_i - x_j)$. The *exact correlation energy* then comes free, with no QMC:

$$
E_c = E_\text{exact} - E_\text{HF}
$$

**M4. LDA exchange for the 1D soft-Coulomb gas (derive it yourselves).** Uniform unpolarized gas with Fermi wavevector $k_F = \pi n/2$ and one-body density matrix

$$
\gamma_\sigma(x) = \frac{k_F}{\pi}\,\frac{\sin(k_F x)}{k_F x}
$$

Inserting into $E_x = -\tfrac12\sum_\sigma\iint w(x-x')\,|\gamma_\sigma(x,x')|^2$ gives the exchange energy density

$$
n\,e_x(n) = -\frac{n^2}{2}\,J(k_F),
\qquad
J(k) = \int_0^\infty \frac{\sin^2 t}{t^2\sqrt{t^2+k^2}}\,dt
$$

Tabulate on a log-spaced $n$ grid, build a `CubicSpline` in $\ln n$, and take

$$
v_x(n) = \frac{d\,[n\,e_x(n)]}{dn}
$$

from the spline derivative. Check: $e_x \to -\tfrac12$ as $n \to \infty$.

**M5. LDA correlation.** The parametrization for exactly this interaction is in Helbig, Fuks, Casula, Verstraete, Marques, Tokatly, Rubio, Phys. Rev. A 83, 032503 (2011), arXiv:1101.2564 (QMC-based fit, $a=1$, unpolarized and polarized). Transcribe the coefficients from Table I of the paper. In 1D,

$$
r_s = \frac{1}{2n}
$$

Validate by reproducing the 1D He energy they report. Fallback if this stalls: drop LDA-C; the project still works with LDA-X against exact, since the exact $E_c$ comes from M3.

**M6. Exact xc potential (no iteration needed for $N=2$).** For a singlet pair, $\varphi = \sqrt{n/2}$ exactly, so the KS equation can be solved for the potential directly:

$$
v_s(x) = \varepsilon + \frac{\varphi''(x)}{2\,\varphi(x)},
\qquad
v_{xc} = v_s - v_\text{ext} - v_H,
\qquad
v_c = v_{xc} + \frac{v_H}{2}
$$

- Fix the constant $\varepsilon$ so $v_{xc} \to 0$ at large $|x|$. In exact DFT, $\varepsilon = -I$.
- Restrict to the region $n \gtrsim 10^{-8}$, because $\varphi''/\varphi$ amplifies noise in the tails.
- Plot exact $v_{xc}$ against the LDA $v_x$. The LDA potential decays exponentially; the exact one decays as $-1/|x|$.
- Optional method comparison: Zhao–Morrison–Parr iteration $v_s \leftarrow v_s + \alpha\,(n_\text{KS} - n_\text{exact})$ (slow and noisy), or Wu–Yang. Use it only to cross-check the direct formula.

### Reference numbers (from my own calculation, $h = 0.1$ with Richardson extrapolation; reproduce before trusting)

| Quantity | Value (Ha) |
|---|---|
| 1D H atom ($Z=1$, one electron) | $-0.6698$ |
| 1D He$^+$ ($Z=2$, one electron) | $-1.4834$ |
| 1D He, exact | $-2.2383$ |
| 1D He, RHF (= KS-EXX) | $-2.2242$ |
| Exact correlation energy $E_c$ | $-0.0140$ |
| Exact ionization energy $I$ | $0.7548$ |
| 1D He, LDA exchange-only | $-2.158$ |
| $E_x$: LDA vs exact ($-E_H/2$) | $-0.654$ vs $-0.724$ |

LDA-X underestimates exchange by about 10%, while correlation is only about 0.014 Ha. So in 1D He most of the LDA error is exchange. I recall the H and He values matching Helbig et al. and Wagner et al. (about $-0.670$ and $-2.238$), but check the paper tables.

### Extension: stretched H$_2$ (strongest extension)

Two soft nuclei at $\pm R/2$ ($Z=1$ each) with nuclear repulsion $w(R)$ added. Grow the box with $R$ ($L = R/2 + 9$ worked). My results ($h = 0.15$, accuracy a few $10^{-4}$ Ha):

| $R$ | Exact | RHF (= KS-EXX) | LDA-X only |
|---|---|---|---|
| 1 | $-1.4134$ | $-1.3947$ | $-1.3331$ |
| 2 | $-1.4421$ | $-1.4060$ | $-1.3561$ |
| 4 | $-1.3509$ | $-1.1904$ | $-1.1281$ |
| 6 | $-1.3395$ | $-1.0693$ | $-1.0729$ |
| 10 | $-1.3398$ | $-1.0224$ | $-1.0651$ |

- The exact curve has a minimum near $R \approx 2$ and dissociates to $2\times(-0.6698) = -1.3396$ Ha.
- RHF dissociates about 0.32 Ha too high. The exact $|E_c|$ grows from 0.015 at $R=0.5$ to 0.317 at $R=10$. This is static correlation: the restricted wavefunction forces 50% ionic character at any $R$.
- LDA-X also fails by about 0.27 Ha at $R=10$. I have not run full LDA-XC; expect the same failure, and treat it as a test of the claim that a better xc functional does not fix static correlation error.
- Plot exact $v_{xc}(x)$ at $R = 1, 4, 8$. At large $R$ a peak/step develops at the bond midpoint, which no LDA reproduces. This is the highlight figure.

### Parameter studies (pick two or three)

1. Grid convergence in $h$ and $L$ (mandatory; it gives your error bar).
2. SCF mixing $\alpha$ versus iteration count, including divergence near $\alpha = 1$.
3. $Z$-scan for two-electron ions: H$^-$, He, Li$^+$, Be$^{2+}$. Plot LDA-X relative error and $|E_c|$ against $Z$ (high-density limit).
4. Softening parameter $a$ in $w(x) = 1/\sqrt{x^2+a^2}$, beyond $a=1$.
5. HOMO test: exact $-\varepsilon_\text{HOMO} = I = 0.7548$ Ha. LDA should give a smaller value because of the wrong asymptotic decay.

### Pitfalls

- Wrong sign or scale on $v_x$: check $E_x^\text{LDA}$ against $-E_H/2$.
- Energy double counting: use $E = T_s + \int n\,v_\text{ext} + E_H + E_{xc}$, not a sum of eigenvalues alone.
- Box too small at large $R$: if the energy drifts as $L$ grows, enlarge it.
- Memory: $N_x^2$ grows fast; $R = 10$ is about 70k unknowns, still fine.
- Scope: finish M1–M3 first. That alone is a complete, defensible project before touching M4–M6.

### Resources

- Capelle, arXiv:cond-mat/0211443.
- Helbig et al., PRA 83, 032503 (2011), arXiv:1101.2564 (LDA parametrization, 1D He benchmark).
- Wagner, Stoudenmire, Burke, White, Phys. Chem. Chem. Phys. 14, 8581 (2012), "Reference electronic structure calculations in one dimension". Check which interaction they use before copying numbers; it may differ from $1/\sqrt{x^2+1}$.
- Cohen, Mori-Sánchez, Yang, Science 321, 792 (2008): delocalization and static correlation error.
- Burke, J. Chem. Phys. 136, 150901 (2012).
- Wu and Yang, J. Chem. Phys. 118, 2498 (2003): potential inversion.

---

## D3. Band structure by empirical pseudopotential (Si, Ge, GaAs)

**What you build.** The band structure of silicon (why it is a semiconductor with a gap of about 1.1 eV, indirect) by diagonalizing a matrix at each $\mathbf k$-point in a plane-wave basis. This is not DFT, but it is the same plane-wave machinery used by Quantum ESPRESSO and VASP.

**Physics.** For a periodic potential $V(\mathbf r) = \sum_{\mathbf G} V(\mathbf G)\,e^{i\mathbf G\cdot\mathbf r}$, Bloch's theorem gives $\psi_{\mathbf k}(\mathbf r) = \sum_{\mathbf G} c_{\mathbf G}\,e^{i(\mathbf k+\mathbf G)\cdot\mathbf r}$. For each $\mathbf k$, solve

$$
\sum_{\mathbf G'}\left[\frac{|\mathbf k+\mathbf G|^2}{2}\,\delta_{\mathbf G\mathbf G'} + V(\mathbf G-\mathbf G')\right]c_{\mathbf G'} = E\,c_{\mathbf G}
$$

**Why computation.** A periodic potential with a two-atom basis has no closed-form band structure. Nearly-free-electron theory works only perturbatively and fails to give quantitative gaps.

**Ingredients**

1. FCC lattice ($a = 5.43$ Å for Si); the reciprocal lattice is BCC. Generate all $\mathbf G$ vectors up to a cutoff (about 100–250 plane waves), in units of $2\pi/a$.
2. Potential matrix from a few form factors, with $\Delta\mathbf G = \mathbf G - \mathbf G'$ and $\boldsymbol\tau = \frac a8(1,1,1)$:
$$
V(\Delta\mathbf G) = V_S(|\Delta\mathbf G|^2)\cos(\Delta\mathbf G\cdot\boldsymbol\tau) + i\,V_A(|\Delta\mathbf G|^2)\sin(\Delta\mathbf G\cdot\boldsymbol\tau)
$$
   For Si only $|\Delta\mathbf G|^2 = 3, 8, 11$ (in units of $(2\pi/a)^2$) contribute and $V_A = 0$. Cohen–Bergstresser Si values are about $V_3 = -0.21$, $V_8 = +0.04$, $V_{11} = +0.08$ Ry. Check these in the paper; units are Rydberg.
3. Loop $\mathbf k$ along $\Gamma$–X–W–L–$\Gamma$–K, diagonalize with `numpy.linalg.eigh`, plot the lowest 8 bands. Convert Ry to Ha or eV carefully ($1\ \text{Ry} = 13.606$ eV).

**Validation.** Si: valence-band maximum at $\Gamma$, conduction-band minimum about 85% of the way along $\Gamma$–X (indirect), gap on the order of 1 eV. GaAs: direct gap at $\Gamma$; needs the antisymmetric form factors.

**Extensions**

- Convergence versus plane-wave cutoff.
- Gap versus lattice constant (pressure dependence, deformation potentials).
- Effective masses from band curvature (finite differences):
$$
\frac{1}{m^*} = \frac{1}{\hbar^2}\frac{\partial^2 E}{\partial k^2}
$$
- Density of states by Gaussian smearing or the tetrahedron method.
- Si vs Ge vs GaAs comparison.
- Optional bridge to real DFT: compare with a GPAW or Quantum ESPRESSO band structure.

**Assessment.** Lowest risk and runs in minutes on a laptop. Weakness: it is a standard textbook exercise, so originality has to come from the extensions.

**Resources**

- Cohen and Bergstresser, Phys. Rev. 141, 789 (1966).
- Chelikowsky and Cohen, Phys. Rev. B 14, 556 (1976).
- Yu and Cardona, *Fundamentals of Semiconductors* (pseudopotential chapter).
- Kittel, *Introduction to Solid State Physics* (Bloch theorem, nearly-free-electron model).

---

## D4. Hartree–Fock and DFT for H$_2$ and HeH$^+$ in a Gaussian basis

**What you build.** The textbook Roothaan SCF procedure with a minimal Gaussian basis, a dissociation curve, and a demonstration of restricted Hartree–Fock's failure for H$_2$. The extension adds an LDA exchange–correlation term on a numerical grid as a bridge to DFT.

**Method.**

- Expand orbitals in STO-3G (each Slater function as 3 Gaussians). All integrals for $s$-type Gaussians have closed forms using the Boys function
$$
F_0(t) = \frac12\sqrt{\frac{\pi}{t}}\,\mathrm{erf}\!\left(\sqrt t\right)
$$
- Solve the generalized eigenproblem (Roothaan equations) and iterate to self-consistency:
$$
\mathbf F\,\mathbf C = \mathbf S\,\mathbf C\,\boldsymbol\varepsilon
$$
- Szabo and Ostlund chapter 3 and Appendix A give the integral formulas.

**Validation.** H$_2$ at $R = 1.4$ bohr, STO-3G: $E \approx -1.1167$ Ha. HeH$^+$ in STO-3G near $R = 1.46$ bohr gives about $-2.86$ Ha; check both against Szabo and Ostlund and against PySCF (which should agree to about $10^{-6}$).

**Extensions**

- Plot $E(R)$. RHF H$_2$ dissociates to the wrong limit (several tenths of a Ha too high, from spurious ionic terms). Note that RHF HeH$^+$ dissociates correctly to He + H$^+$, which is an instructive contrast.
- Add an LDA xc term via numerical integration on a molecular grid (the DFT bridge); compare to the RHF curve.
- Optional: unrestricted HF, which fixes the dissociation limit at the cost of spin contamination.

**Assessment.** Classic, well documented, low risk, but it is a rederivation of textbook material. The two-electron-integral bookkeeping is where beginners get stuck.

**Resources**

- Szabo and Ostlund, *Modern Quantum Chemistry*, ch. 3 and Appendix A.
- Crawford Group Programming Projects (github.com/CrawfordGroup/ProgrammingProjects).
- PySCF docs (pyscf.org) for validation.

---

## Recommendation

- **D2 as the main project**, because the exact reference makes the error analysis quantitative and the stretched-H$_2$ extension is a strong "limitations of the method" argument.
- **D1 as the lower-risk alternative** (published targets at every step), or as a validation/extension block if time allows.
- D3 is the safety net if the group dislikes SCF. D4 is the most textbook-like.
- Suggested split for 3 people: one on the solver(s), one on the KS/SCF side, one on reference runs, plotting and the slide skeleton from day 1.