# 1D soft-Coulomb two-electron system: exact vs Kohn–Sham

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
