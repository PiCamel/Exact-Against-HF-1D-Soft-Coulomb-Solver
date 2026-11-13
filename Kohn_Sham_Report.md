# Kohn–Sham DFT on a 1D helium atom: theory, methods and results

Nov 13, 2026 · @Kushal

I rebuilt your project end to end, and every atom number reproduces your reference table to four decimals: exact He = −2.2383 Ha, Hartree–Fock = −2.2242 Ha, correlation energy = −0.0140 Ha, ionization energy = 0.7548 Ha. The comparison shows LDA exchange missing about 10% of the exchange energy, and restricted Hartree–Fock failing badly once H2 is stretched.

One exception: my stretched-H2 RHF and LDA-X energies for R of 4 and above are lower than the table in your notebook. I checked mine against an independent variational calculation (section C4).

Part A teaches the theory from scratch, B gives the methods, C the results, D the bugs and fixes in your notebook, and E what to do next. All numbers come from the corrected code in Kohn\_Sham\_sim\_fixed.ipynb.

## A1. The problem DFT solves

An atom is a nucleus plus electrons. To get its energy we need the wavefunction Ψ, which encodes the probability of finding the electrons at given positions.

The trouble is size. For N electrons in 3D, Ψ is a function of 3N coordinates. With just 10 grid points per coordinate you must store 10^(3N) numbers, which is 10^30 for N = 10. No computer can do that.

Density functional theory (DFT) sidesteps this. It works with the electron density n(x), a function of one position only, however many electrons there are. The catch is that the exact formula for the energy in terms of n is unknown, so we approximate it.

Your project measures how good those approximations are. It does so in a toy world small enough that the exact answer can be computed, so every error is visible.

## A2. The toy world

Two electrons live on a line (1D). A nucleus of charge Z sits at x = 0. We use atomic units: lengths in bohr, energies in hartree (Ha), and ħ = m = e = 1, so the constants disappear from every equation.

$$
\hat H = -\tfrac12\partial_{x_1}^2 - \tfrac12\partial_{x_2}^2 - Z\,w(x_1) - Z\,w(x_2) + w(x_1-x_2)
$$

The first two terms are the kinetic energy of each electron. The next two are the attraction of each electron to the nucleus. The last is the repulsion between the two electrons.

Every Coulomb interaction is softened:

$$
w(x) = \frac{1}{\sqrt{x^2+1}}
$$

Far away this looks like the real 1/|x|. Near x = 0 it stays finite (w = 1) instead of blowing up. Pure 1/|x| in one dimension is pathological, so the softening is what makes the problem well behaved.

Changing Z gives different atoms: Z = 1 is the negative ion H⁻ (two electrons, one proton), Z = 2 is helium, Z = 3 is Li⁺.

**Spin and symmetry.** Electrons obey the Pauli principle: the total wavefunction flips sign when you swap the two electrons. Our Hamiltonian has no spin in it, so Ψ splits into a position part times a spin part.

The lowest-energy position function has no nodes (no places where it crosses zero), so it is symmetric under x1 ↔ x2. Symmetric position part forces the antisymmetric spin singlet, so the two electrons have opposite spins. We solve for one real, positive, symmetric function Ψ(x1, x2).

**Density.** Integrating out one electron gives the probability per unit length of finding an electron at x. The factor 2 counts either electron:

$$
n(x) = 2\int |\Psi(x,x')|^2\,dx', \qquad \int n(x)\,dx = 2
$$

Always check that the second equation holds numerically. It catches normalisation mistakes immediately.

## A3. Solving the exact problem on a grid (M1 and M2)

**Step 1: a grid.** Replace the continuous line by points x\_i = −L + i·h, spaced h apart, from −L to L. A function ψ(x) becomes a list of numbers ψ\_i. Here L = 10 and h = 0.1 give N = 201 points.

**Step 2: derivatives become differences.** Taylor-expand ψ(x+h) and ψ(x−h), add them, and the first-derivative terms cancel. What remains is the three-point formula:

$$
\psi''(x_i) \approx \frac{\psi_{i+1} - 2\psi_i + \psi_{i-1}}{h^2}
$$

The leftover error is proportional to h squared. Halving h cuts the error by a factor of 4.

**Step 3: a matrix eigenvalue problem.** Plug the formula into the Schrödinger equation. The operator −½ d²/dx² becomes a matrix with 1/h² on the diagonal and −1/(2h²) just beside it. Adding the potential v(x\_i) on the diagonal gives

$$
H\boldsymbol\psi = E\boldsymbol\psi,\quad H_{ii} = \frac{1}{h^2} + v(x_i),\quad H_{i,i\pm1} = -\frac{1}{2h^2}
$$

This is your kinetic\_matrix function. The ground state is the eigenvector with the lowest eigenvalue. The reason is the variational principle: any trial vector gives an energy at or above the true ground-state energy.

$$
E[\psi] = \frac{\langle\psi|\hat H|\psi\rangle}{\langle\psi|\psi\rangle} \ge E_0
$$

So asking for the smallest eigenvalue (which='SA' in eigsh, meaning smallest algebraic) is the same as asking for the most stable state. That is the correct explanation of the comment in your notebook.

**Step 4: normalisation.** A numerical eigenvector has a sum of squares equal to 1. A physical wavefunction needs the integral of ψ² equal to 1, and an integral is a sum times h. So ψ = vector/√h in 1D, and Ψ = vector/h in 2D (your Psi = P.reshape(N,N)/h).

**Step 5: two electrons.** Now Ψ(x1, x2) is a table with N rows (electron 1) and N columns (electron 2). Flatten it into one vector of N² entries, using index k = i·N + j.

The Kronecker product builds the operator on that flat vector. T⊗I applies the kinetic energy to electron 1 and I⊗T applies it to electron 2:

$$
H_{2e} = T\otimes I + I\otimes T + \mathrm{diag}(V)
$$

The potential V is a diagonal matrix. Its entry for the pair (x\_i, x\_j) is v(x\_i) + v(x\_j) + w(x\_i − x\_j). Your code builds it with three lines that are worth understanding:

- meshgrid(x, x, indexing='ij') makes two tables with X1\[i,j\] = x\_i and X2\[i,j\] = x\_j.
- w(X1 − X2).ravel() is then the electron–electron repulsion for every pair, flattened row by row.
- np.repeat(v, N) gives v(x\_i) at index k = i·N + j, and np.tile(v, N) gives v(x\_j). Together they are the two nuclear attractions.

For N = 201 the matrix has 40,401 rows but only about 5 non-zero entries per row, so we store it sparse. The solver eigsh (Lanczos method) finds the lowest eigenvalue using only matrix–vector products, which is why it runs in seconds.

**Step 6: the density.** Sum Ψ² over the second electron's position: n\_i = 2·h·Σ\_j Ψ\_ij².

**Step 7: Richardson extrapolation.** Because the error is c·h², two runs at h and 2h can be combined to cancel it:

$$
E(h) \approx E_0 + c\,h^2 \;\Rightarrow\; E_0 \approx E(h) + \frac{E(h) - E(2h)}{3}
$$

Check it: E(h) − E(2h) = −3c·h², so dividing by 3 gives −c·h², which cancels the error in E(h).

&#91;image: Exact two-electron wavefunction of 1D helium on the (x1, x2) plane\]

*Figure 1. The exact wavefunction Ψ(x1, x2) of 1D helium. It is positive and symmetric about the diagonal x1 = x2.*

## A4. Hartree–Fock and the self-interaction idea (M3)

The cheapest wavefunction guess puts both electrons (opposite spins) in the same position orbital φ(x). Then Ψ(x1, x2) = φ(x1)φ(x2), and the energy is

$$
E[\varphi] = 2\langle\varphi|\hat T + v|\varphi\rangle + \iint \varphi(x)^2\,w(x-x')\,\varphi(x')^2\,dx\,dx'
$$

The last term is the repulsion of two charge clouds. It is built from the Hartree potential, the electrostatic potential of the whole electron cloud n = 2φ²:

$$
v_H(x) = \int n(x')\,w(x-x')\,dx', \qquad E_H = \tfrac12\int n\,v_H\,dx
$$

In code, the integral is a matrix–vector product: v\_H = W @ n \* h, where W\[i,j\] = w(x\_i − x\_j). The factor h is the integration weight. Your notebook sketch is missing it.

**Self-interaction.** The cloud n contains both electrons, so each electron feels repulsion from its own cloud. That is unphysical. For two electrons in one orbital, exactly half of E\_H is this spurious self-repulsion, and exchange removes it:

$$
E_x = -\tfrac12 E_H, \qquad E_{HF} = 2T_\varphi + \int n\,v_{ext} + E_H + E_x
$$

Minimising E\[φ\] with the constraint that φ stays normalised gives an eigenvalue equation for φ. In it each electron sees the nucleus plus only half the Hartree potential:

$$
\Big[-\tfrac12\frac{d^2}{dx^2} + v_{ext}(x) + \tfrac12 v_H(x)\Big]\varphi = \varepsilon\,\varphi, \qquad n = 2\varphi^2
$$

The potential depends on φ and φ depends on the potential, so we solve by iteration (section A5). Because the one-orbital picture ignores how electrons dodge each other, it misses some energy. That missing piece is the correlation energy, and in this project it comes free:

$$
E_c = E_{exact} - E_{HF}
$$

## A5. DFT and the Kohn–Sham trick

**Hohenberg–Kohn (1964).** The ground-state density n(x) determines the external potential, and therefore everything, including the energy. There is a universal functional F\[n\], the same for every atom and molecule, such that

$$
E[n] = F[n] + \int n(x)\,v_{ext}(x)\,dx \;\ge\; E_0
$$

The true ground-state density minimises E\[n\]. So in principle we can find the energy by minimising over densities instead of wavefunctions.

**The catch.** F\[n\] contains the kinetic energy, and nobody knows its exact form as a function of n. Guessed formulas for it are poor.

**The Kohn–Sham trick (1965).** Invent a fictitious system of non-interacting electrons that is arranged to have the same density as the real one. Non-interacting electrons have an exactly computable kinetic energy T\_s, from their orbitals. Everything else is lumped into one unknown piece, the exchange–correlation energy E\_xc:

$$
E[n] = T_s[n] + \int n\,v_{ext}\,dx + E_H[n] + E_{xc}[n]
$$

E\_xc holds the exchange, the correlation, and the small difference between the true and fictitious kinetic energies. If we knew E\_xc exactly, DFT would give the exact energy. All approximation in DFT lives in E\_xc.

Minimising gives the Kohn–Sham equations. Each fictitious electron moves in an effective potential, and the density is the sum of orbital probabilities (times 2 for the two spins):

$$
\Big[-\tfrac12\frac{d^2}{dx^2} + v_{ext} + v_H + v_{xc}\Big]\varphi_i = \varepsilon_i\varphi_i, \qquad v_{xc} = \frac{\delta E_{xc}}{\delta n}
$$

The symbol δ/δn means the functional derivative: how much E\_xc changes when n changes slightly at one point. The orbitals φ\_i are not real electrons. Only their density is meant to be physical.

**Warning: energy is not the sum of eigenvalues.** Adding the ε\_i counts the electron–electron interaction twice and uses v\_xc where the energy needs E\_xc. Use the energy functional itself, or the corrected sum (over electrons):

$$
E = \sum_i \varepsilon_i - E_H - \int n\,v_{xc}\,dx + E_{xc}
$$

**Link to Hartree–Fock.** For two electrons in one orbital, E\_x = −E\_H/2 is exact, and v\_x = δE\_x/δn = −v\_H/2. So KS with exact exchange and no correlation is the same as restricted Hartree–Fock. That is why M3 gives both at once.

**The self-consistent field (SCF) loop.** The potential depends on n and n depends on the potential, so we iterate:

1. Guess n (the bare-nucleus orbital is a fine start).
2. Build v\_H and v\_xc, then solve the 1D eigenproblem for the lowest orbital.
3. Form the new density n\_out = 2φ².
4. Mix, so the next input is a blend of old and new.
5. Stop when the density change is tiny.

$$
n_{in}^{(k+1)} = \alpha\,n_{out}^{(k)} + (1-\alpha)\,n_{in}^{(k)}, \qquad \|n_{out}-n_{in}\| = \sqrt{\textstyle\sum_i (n_{out,i}-n_{in,i})^2\,h}
$$

Why mix? Think of aiming a thermostat: if the output overshoots the input each time, the iteration swings wider and wider. A fraction α of the new density damps the swing. Small α is safe but slow, large α is fast but can diverge. Section C3 measures this.

## A6. LDA exchange for the 1D soft-Coulomb gas (M4)

The local density approximation (LDA) says: at each point, pretend the electrons sit in a uniform gas with the same local density n, and borrow that gas's exchange energy. So we need the exchange energy of a uniform 1D gas with our softened interaction. You derive it in four steps.

**Step 1: the gas.** Free electrons on a long line occupy plane waves e^(ikx) with wavevector up to the Fermi wavevector k\_F. Per spin, the number of states per unit length with |k| below k\_F is 2k\_F/(2π) = k\_F/π. Setting that equal to n/2 gives

$$
k_F = \frac{\pi n}{2}
$$

**Step 2: the density matrix.** Add up the plane waves to see how much an electron at one point is correlated with a point a distance x away:

$$
\gamma(x) = \frac{1}{2\pi}\int_{-k_F}^{k_F} e^{ikx}\,dk = \frac{k_F}{\pi}\,\frac{\sin(k_F x)}{k_F x}
$$

**Step 3: the exchange energy.** Exchange is minus one half of the interaction weighted by γ², summed over the two spins. For a uniform gas this gives an energy per unit length of

$$
n\,e_x(n) = -\int_{-\infty}^{\infty} w(x)\,\gamma(x)^2\,dx
$$

Now substitute t = k\_F·x. The softened interaction becomes w = k\_F/√(t² + k\_F²), and the integral turns into

$$
n\,e_x(n) = -\frac{n^2}{2}\,J(k_F), \qquad J(k) = \int_0^\infty \frac{\sin^2 t}{t^2\sqrt{t^2+k^2}}\,dt
$$

**Step 4: numbers.** J has no closed form, so we integrate it numerically. Beyond t = 400 the integrand is smooth (sin² averages to ½), so the tail is added analytically. We tabulate n·e\_x on 500 log-spaced densities from 10⁻¹⁰ to 2000, fit a cubic spline in ln n, and get the potential from its derivative:

$$
E_x^{LDA}[n] = \int n\,e_x(n)\,dx, \qquad v_x(n) = \frac{d\,[\,n\,e_x(n)\,]}{dn}
$$

**Sanity check.** At very high density J ≈ π/(2k\_F), because the factor 1/√(t²+k²) is nearly constant over the region where sin²t/t² lives, and the integral of sin²t/t² is π/2. That gives e\_x → −½. The code returns e\_x(n = 1000) = −0.49990.

&#91;image: LDA exchange energy per electron against density\]

*Figure 2. LDA exchange energy per electron e\_x(n) for the soft-Coulomb gas. It tends to −½ at high density.*

**Why LDA exchange is too weak.** The exact exchange for two electrons is −E\_H/2, which cancels the self-repulsion exactly. A local formula only approximates that cancellation. For He it gives E\_x = −0.654 Ha against −0.724 Ha, about 10% short.

## A7. The exact exchange–correlation potential by inversion (M6)

For two electrons with opposite spins, n = 2φ² means the single Kohn–Sham orbital is exactly φ = √(n/2). No iteration is needed: put that φ into the KS equation and solve for the potential.

$$
v_s(x) = \varepsilon + \frac{\varphi''(x)}{2\,\varphi(x)}, \qquad v_{xc} = v_s - v_{ext} - v_H, \qquad \varepsilon = -I
$$

The constant ε is fixed by a theorem: in exact DFT the highest occupied orbital energy equals minus the ionization energy I. We take I from the two exact energies, E(He⁺) − E(He).

Two things to know. First, φ''/φ divides tiny numbers in the tails, so noise explodes there. We only plot where n > 10⁻⁸. Second, far from the atom the escaping electron sees the nucleus (+Z) screened by the other electron (−1), a net +1. So the exact v\_xc must tend to −w(x), which decays as 1/|x|. LDA gives something proportional to n, which dies exponentially.

## A8. Why stretching H2 breaks things

Place two nuclei (Z = 1 each) at ±R/2, and add the nuclear repulsion w(R) to every energy. Restricted Hartree–Fock still forces both electrons into one orbital. By symmetry that orbital is φ = (χ\_A + χ\_B)/√2, a mix of an atomic orbital on each nucleus.

Multiply it out and the two-electron wavefunction has four terms:

$$
\Psi_{RHF} = \tfrac12\big[\chi_A(x_1)\chi_A(x_2) + \chi_B(x_1)\chi_B(x_2) + \chi_A(x_1)\chi_B(x_2) + \chi_B(x_1)\chi_A(x_2)\big]
$$

The last two terms have one electron on each atom (neutral H + H). The first two put both electrons on the same atom (H⁻ + H⁺). So RHF is 50% ionic at every R, even when the atoms are far apart, where the ionic state costs a lot of energy.

This is static (strong) correlation: the true state needs the covalent configuration alone, and a single orbital cannot do that. The exact exchange–correlation potential fights it by growing a peak at the bond midpoint, which holds the electrons on separate atoms. A local functional like LDA cannot build that peak, so it fails the same way.

## B. Methods

- **Grid.** Uniform grid on \[−L, L\]. Atoms: L = 10, h = 0.1, with h = 0.2 for Richardson extrapolation. Stretched H2: L = R/2 + 9, h = 0.15.
- **Exact solver.** Sparse 2D Hamiltonian (Kronecker products), lowest states from eigsh with tolerance 1e-10. Four states for He to find the triplet.
- **SCF (RHF and LDA-X).** Start from the bare-nucleus orbital. Mix with α = 0.5 for atoms. For H2 use α = 0.2 and symmetrise the density under x → −x each step, which keeps the σ\_g-like solution. Stop when the L2 density change falls below 1e-9.
- **Energies.** E = T\_s + ∫n v\_ext + E\_H + E\_xc + w(R). Never the sum of eigenvalues.
- **LDA exchange.** J(k) by adaptive quadrature plus analytic tail, cubic spline in ln n, v\_x from the spline derivative. No LDA correlation (M5 not done, see section E).
- **Exact v\_xc.** Direct inversion, ε = −I from same-grid energies, shown where n > 10⁻⁸.
- **Validation.** ∫n = 2 for every exact density; E\_x(LDA) against −E\_H/2; the RHF energy re-evaluated as the expectation value of the full two-electron Hamiltonian; Richardson and box-size tests.

## C. Results

### C1. Atoms: every reference number reproduces

| Quantity (Ha) | This work | Your notebook's table |
| --- | --- | --- |
| 1D H atom (Z = 1, one electron) | −0.66978 | −0.6698 |
| 1D He⁺ (Z = 2, one electron) | −1.48343 | −1.4834 |
| 1D He, exact | −2.23826 | −2.2383 |
| 1D He, RHF = KS-EXX | −2.22421 | −2.2242 |
| Exact correlation energy E\_c | −0.01405 | −0.0140 |
| Exact ionization energy I | 0.75483 | 0.7548 |
| 1D He, LDA exchange only | −2.15819 | −2.158 |
| E\_x: LDA vs exact (−E\_H/2, at the RHF density) | −0.6535 vs −0.7239 | −0.654 vs −0.724 |

All values use h = 0.1 and h = 0.2 with Richardson extrapolation, L = 10.

The LDA-X energy error is 0.080 Ha, which is 5.7 times the whole correlation energy of 0.014 Ha. So in 1D He most of the LDA error is exchange, not correlation.

&#91;image: Density of 1D helium: exact, Hartree-Fock and LDA-X\]

*Figure 3. Ground-state density of 1D He. All three curves nearly coincide, with HF slightly too peaked at the nucleus. The density looks right even though the LDA-X energy is off by 0.08 Ha.*

Extra results from the same runs:

- **HOMO test.** The exact −ε\_HOMO equals I = 0.7548 Ha. RHF gives 0.7503 Ha, close. LDA-X gives only 0.4673 Ha, because its potential decays too fast.
- **Triplet.** The lowest antisymmetric state is at −1.8164 Ha, so the singlet–triplet gap is 0.4221 Ha.
- **H⁻ (Z = 1).** The exact energy is −0.7303 Ha, below the neutral atom's −0.6698 Ha. So the ion is bound, with electron affinity near 0.06 Ha. This value is still box-limited (see C2).

### C2. Convergence: error scales as h squared

| h (bohr) | E exact He (Ha) |
| --- | --- |
| 0.4 | −2.24302 |
| 0.2 | −2.23943 |
| 0.1 | −2.23855 |
| 0.05 | −2.23833 |
| extrapolated | −2.23826 |

Successive differences shrink by factors of 4.07 and 4.02, which is the O(h²) behaviour from section A3. For the box: with L = 5 the He energy is off by 3.8e-4 Ha, with L = 7 by 5e-6 Ha, and L = 10 is converged.

H⁻ is far more diffuse. Its energy still moves by 3.6e-4 Ha between L = 10 and L = 14, so ion results at L = 10 carry that uncertainty.

&#91;image: Grid-spacing and box-size convergence\]

*Figure 4. Left: error against h follows h squared. Right: error against box half-width L, for He and H⁻.*

### C3. SCF mixing for helium (RHF)

Iterations to reach 1e-8: 138 at α = 0.1, 66 at 0.2, 41 at 0.3, 29 at 0.4, 21 at 0.5, 16 at 0.6, 12 at 0.7, 9 at 0.8, 7 at 0.9, and 10 at α = 1.0. Even plain iteration converges for He, so the divergence warning in your notes does not show up here.

It does show up for stretched H2. At α = 0.5 the SCF never converges for R of 4 and above. At α = 0.1 it converges for R up to 6, and for R of 8 and above it hops to a symmetry-broken, one-sided density unless the density is symmetrised.

&#91;image: SCF iterations against mixing parameter\]

*Figure 5. RHF iterations for He against the mixing parameter α.*

### C4. Stretched H2: static correlation

| R (bohr) | Exact (Ha) | RHF (Ha) | LDA-X (Ha) | RHF − exact | LDA-X − exact |
| --- | --- | --- | --- | --- | --- |
| 0.5 | −1.3121 | −1.2969 | −1.2320 | 0.0152 | 0.0801 |
| 1 | −1.4134 | −1.3947 | −1.3331 | 0.0187 | 0.0803 |
| 2 | −1.4421 | −1.4060 | −1.3561 | 0.0361 | 0.0860 |
| 3 | −1.3850 | −1.3115 | −1.2770 | 0.0735 | 0.1080 |
| 4 | −1.3509 | −1.2233 | −1.2060 | 0.1275 | 0.1448 |
| 6 | −1.3395 | −1.1207 | −1.1393 | 0.2188 | 0.2002 |
| 8 | −1.3397 | −1.0771 | −1.1237 | 0.2626 | 0.2159 |
| 10 | −1.3398 | −1.0572 | −1.1208 | 0.2826 | 0.2191 |

The exact curve has its minimum near R = 2, at −1.4421 Ha, and dissociates to −1.3398 Ha, matching two H atoms (−1.3396 Ha). The RHF error grows from 0.015 Ha at R = 0.5 to 0.283 Ha at R = 10, which is the exact correlation energy magnitude there. LDA-X fails by 0.22 Ha at R = 10.

LDA-X looks slightly better than RHF at R of 6 and above. That is probably error cancellation between its weak exchange and the missing static correlation, not a sign of a better functional.

&#91;image: Stretched H2 energy curves\]

*Figure 6. Stretched 1D H2. Exact dissociates to two H atoms; RHF and LDA-X stay far too high.*

**Where my numbers differ from your table.** The exact column matches yours at every R. The RHF and LDA-X columns match at R = 1 and 2 but not beyond:

| R | RHF: your table | RHF: mine | LDA-X: your table | LDA-X: mine |
| --- | --- | --- | --- | --- |
| 4 | −1.1904 | −1.2233 | −1.1281 | −1.2060 |
| 6 | −1.0693 | −1.1207 | −1.0729 | −1.1393 |
| 10 | −1.0224 | −1.0572 | −1.0651 | −1.1208 |

I verified mine: at R = 4 the SCF energy −1.223314 Ha equals the expectation value of the full two-electron Hamiltonian in the product state φ(x1)φ(x2). RHF is a variational upper bound, so a lower value means a better-converged restricted state. Your table's larger values likely come from a less converged or different SCF solution. This also changes the exact |E\_c| at R = 10 from 0.317 Ha to 0.283 Ha. Re-run your original SCF before quoting the table.

### C5. The exchange–correlation potential

For He, the exact v\_xc follows −w(x) in the tail: −0.170 Ha at x = 6 against −w = −0.164, while the LDA exchange potential is already −0.0003 there.

&#91;image: Exchange-correlation potential of 1D helium\]

*Figure 7. Exact v\_xc for He against EXX and LDA. The exact and EXX potentials follow −w(x) in the tail, while LDA decays exponentially.*

For stretched H2, the exact v\_xc at the bond midpoint is −0.756 Ha at R = 1, −0.310 Ha at R = 4 and +0.179 Ha at R = 8. The growing peak between the two wells is the signature of static correlation, and no LDA reproduces it.

&#91;image: Exchange-correlation potential of stretched H2\]

*Figure 8. Exact v\_xc of H2 at R = 1, 4 and 8. A peak grows at the bond midpoint as the bond stretches.*

### C6. Two-electron ions (Z scan, h = 0.1, L = 10)

| Z | Exact (Ha) | RHF (Ha) | E\_c (Ha) | E\_x(LDA) / E\_x(HF) | LDA-X − exact (Ha) |
| --- | --- | --- | --- | --- | --- |
| 1 (H⁻) | −0.73043 | −0.69182 | −0.0386 | 0.879 | 0.0863 |
| 2 (He) | −2.23855 | −2.22450 | −0.01405 | 0.903 | 0.0801 |
| 3 (Li⁺) | −3.89630 | −3.88815 | −0.0082 | 0.904 | 0.0803 |
| 4 (Be²⁺) | −5.61598 | −5.61039 | −0.0056 | 0.905 | 0.0809 |

As Z grows the electrons get denser and correlation shrinks. The LDA exchange shortfall stays near 10% and the absolute energy error stays near 0.08 Ha. The H⁻ row is box-limited.

## D. Bugs and fixes in your notebook

The exact solver (M2) is essentially right. The problems are consistency, missing checks, and an unfinished KS section. All fixes are applied in Kohn\_Sham\_sim\_fixed.ipynb.

| Where | Problem | Fix |
| --- | --- | --- |
| Cell 8 (H2) | Electron–electron softening is 0.2, but the nuclei use the default 1.0. Two different models sit in one Hamiltonian. | One softening parameter a, passed to every interaction. |
| Cell 8 | a = 0.2 with h = 0.1 puts only two grid points across the scale of the interaction. | Keep h at most about a/4, and test convergence in h. |
| Cell 8 | No nuclear repulsion w(R), so energies at different R are not comparable. | Add w(R) to every energy. |
| Cells 5 and 8 | H\_two\_electron(x, 0.1, …) hard-codes h, while x comes from L and N separately. A mismatch silently scales every energy. | grid(L, h) returns x and h = x\[1\] − x\[0\]; pass that h everywhere. |
| Cells 5 and 8 | The energy is never printed or checked. No normalisation test, no Richardson. L = 10 is too small for H⁻ (error about 4e-4 Ha, still drifting). | Print E and the integral of n, extrapolate in h, and test L. |
| Cells 5, 8, 17 | The names n, E, Psi are reused, so later KS cells can compare against the wrong data. | Distinct names such as n\_exact and n\_ks. |
| Cell 15 | W\_grid contains x\[None:,\], which works only by accident. It was meant to be x\[None,:\]. N and h arguments are unused. | W = w(x\[:, None\] − x\[None, :\]). |
| Cell 17 | Computes the whole spectrum, then picks the minimum, and calls np.conj on a real orbital. | select='i', select\_range=(0,0) for the lowest state only; n = 2·φ². |
| Cell 19 | Unfinished: def V\_H(n) has no body. The tolerance comment is ambiguous. | v\_H = W @ n \* h, and a stated L2 norm that includes h. |
| Cell 11 | The KS equation is written in 3D with ħ and m, and the symbols ψ, φ and ρ are mixed. The density sum lacks the factor 2 for spin. | 1D atomic units, symbols n and φ\_i, and n = 2·Σ |
| Cell 6 note | The Kato cusp theorem is about true Coulomb singularities. A soft-Coulomb potential is smooth, so Ψ has no cusp. | Say the density peak sharpens as a shrinks and approaches a cusp only as a goes to 0. |
| Cell 10 note | Metropolis–Hastings does not make an eigenproblem cheaper. Variational Monte Carlo is a different method: stochastic, with an upper bound. | Treat it as a separate comparison, not a shortcut. |

## E. Direction: what to do next

In order of value per hour:

1. **Switch to the fixed notebook and rerun it top to bottom.** Confirm table C1. This also gives you working M3, M4 and M6.
2. **Settle the stretched-H2 discrepancy.** Rerun your original SCF with α of 0.1 or less and symmetrised density, then compare with section C4.
3. **Add LDA correlation (M5).** Transcribe the coefficients from Table I of Helbig et al., with r\_s = 1/(2n) in 1D. Validate by reproducing their 1D He energy. If it stalls, drop it: the exact E\_c is already free from M3, so the project stands without it.
4. **Do two parameter studies.** The mixing study (extend C3 to LDA and to R of 4 and above, and try Pulay/DIIS mixing), and the softening scan a = 0.5, 1, 2 with h at most a/4. The Z scan and the grid test are done.
5. **Extend with symmetry breaking.** Run spin-unrestricted HF and spin-polarised LDA on stretched H2. The energy should then dissociate correctly, at the price of a wrong spin state. That contrast is a good discussion point.
6. **Write the final report in this shape.** Method, exact reference, RHF, LDA, then the H2 extension, with error bars of a few 1e-4 Ha from the h = 0.15 grid.

## F. Self-test

If you can answer these without looking, you own the theory.

1. Why does the grid Hamiltonian have 1/h² on its diagonal? (Three-point second derivative.)
2. Why divide the eigenvector by √h, and by h in 2D? (Sums become integrals.)
3. Why is E\_x = −E\_H/2 for two electrons in one orbital? (It cancels the self-repulsion.)
4. Why is the energy not the sum of KS eigenvalues? (Double counting, and v\_xc is not E\_xc.)
5. Why does RHF fail when H2 is stretched? (Fixed 50% ionic character: static correlation.)

## Sources

These are the references listed in your own notebook. I did not re-open them for this document, so check the details before citing.

- Capelle, arXiv:cond-mat/0211443 (a bird's-eye view of DFT).
- Helbig, Fuks, Casula, Verstraete, Marques, Tokatly, Rubio, Phys. Rev. A 83, 032503 (2011).
- Wagner, Stoudenmire, Burke, White, Phys. Chem. Chem. Phys. 14, 8581 (2012).
- Cohen, Mori-Sánchez, Yang, Science 321, 792 (2008).
- Burke, J. Chem. Phys. 136, 150901 (2012).
- Wu and Yang, J. Chem. Phys. 118, 2498 (2003).
- https://arxiv.org/pdf/1202.4788

All computed numbers in this document are from my own runs of Kohn\_Sham\_sim\_fixed.ipynb on 13 Nov 2026.
