# Exact vs Kohn–Sham DFT for a 1D two-electron system - EP4210 Computational Physics project.

### - By Siddharth Shivakumar (EP25BTECH11027), Kushal R (EP25BTECH11012), and Zulfqar Farooq (EP25BTECH11036)
We solve a two-electron problem **exactly** on a grid, then solve the same problem with **Kohn–Sham density functional theory (DFT)**, and measure where and why the approximate method goes wrong. Because the exact answer is computable here, every error in the approximate method can be measured directly instead of guessed.

The Kohn–Sham calculations use exact exchange, which for two electrons is the same as restricted Hartree–Fock. The local density approximation (LDA), the functional real DFT codes are built on, is discussed in section 5 but not implemented.

All quantities are in Hartree atomic units (ħ = mₑ = e = 1): energies in Hartree (Ha, 1 Ha = 27.211 eV), lengths in bohr.

---

## 1. The model

Two electrons on a line. Every Coulomb interaction is softened so nothing diverges:

```
w(x) = 1 / sqrt(x² + a²),   a = 1

H = -½ ∂²/∂x₁² - ½ ∂²/∂x₂² + v_ext(x₁) + v_ext(x₂) + w(x₁ - x₂)
```

| System | External potential | Notes |
|---|---|---|
| H⁻ | v_ext = -w(x), one nucleus with Z = 1 at x = 0 | Weakly bound: the second electron is held by only about 0.06 Ha |
| He | v_ext = -2w(x), Z = 2 | Tightly bound; the cleanest benchmark |
| H₂ | v_ext = -w(x - R/2) - w(x + R/2), two nuclei | Total energy also includes the nucleus–nucleus repulsion w(R) |

- **Why soften?** The bare 1D Coulomb potential 1/|x| has no finite ground state: the electron collapses onto the nucleus. With a = 1, w(0) = 1 and a plain grid converges. The same a is used for every interaction, so every method describes the same physical system.
- **Spin.** H has no spin operators, so the two-electron ground state is a nodeless, symmetric spatial Ψ(x₁, x₂) paired with the spin singlet.
- **Density.** n(x) = 2 ∫ |Ψ(x, x')|² dx', with ∫ n dx = 2.
- **Born–Oppenheimer.** For H₂ the nuclei are held fixed. Their repulsion w(R) is a constant at fixed R. It doesn't change the wavefunction, but it is part of the total energy at that geometry, and it creates the minimum in the energy curve E(R).

---

## 2. How to run

Requirements: Python 3 with `numpy`, `scipy` and `matplotlib`.

```bash
python compare.py          # runs every solver, prints every number in the report, saves the plots in results/ (about 1.5 minutes)
python KS/KS_H2.py         # any single analysis file runs on its own, prints its energies and plots n(x)
```

---

## 3. Files

```
common.py          shared building blocks, imported by every solver
compare.py         runs every solver, prints all results and makes the comparison figures
exact/             exact two-electron solves (the benchmark)
    exact_H_minus.py   exact_He.py   exact_H2.py
KS/                Kohn–Sham with exact exchange (= restricted Hartree–Fock)
    KS_H_minus.py      KS_He.py      KS_H2.py
results/           plots written by compare.py
```

Each method has one folder, and each folder has one file per system: H⁻ (Z = 1), He (Z = 2) and H₂.

| File | What it does |
|---|---|
| `common.py` | Shared building blocks: the grid, the soft-Coulomb interaction `w`, external potentials, the kinetic-energy matrix, the two-electron Hamiltonian, the Hartree potential and the one-electron solver. |
| `exact/exact_H_minus.py` | Exact two-electron ground state for one nucleus of charge Z (default Z = 1: H⁻). |
| `exact/exact_He.py` | The same for He (Z = 2). |
| `exact/exact_H2.py` | Exact two-electron ground state for H₂ at internuclear distance R. |
| `KS/KS_H_minus.py` | Kohn–Sham with exact exchange for one nucleus (default Z = 1: H⁻). For two electrons this is restricted Hartree–Fock (RHF). |
| `KS/KS_He.py` | The same for He (Z = 2). |
| `KS/KS_H2.py` | Kohn–Sham with exact exchange (= RHF) for H₂. |
| `compare.py` | Runs every solver and saves the energy and density comparison plots in `results/`. |

Every analysis file has a `run(...)` function that returns a dictionary of results (energies, density, grid, convergence history), so `compare.py` can use it, plus a `__main__` block for running it alone. Each solver file adds the project folder to `sys.path` at the top, so it can import `common.py` whether you run it as `python KS/KS_He.py` from the project folder or from inside its own folder.

### `common.py`

- `make_grid(L, h)`: uniform grid from −L to L with spacing h and an odd number of points, so x = 0 is on the grid.
- `w`, `v_1atom`, `v_2atom`, `v_ext`, `v_ext_H2`: interaction and external potentials.
- `kinetic_matrix(N, h)` / `T_1e(N, h)`: −½ d²/dx² as a tridiagonal matrix, from the 3-point finite difference φ″ᵢ ≈ (φᵢ₊₁ − 2φᵢ + φᵢ₋₁)/h². The error is O(h²). Wavefunctions are set to zero outside the box.
- `H_two_electron(x, h, v)`: the two-electron Hamiltonian on the N × N grid, built as `kron(T, I) + kron(I, T) + diag(V)`. Each electron's kinetic energy acts along its own index, so this is the 5-point Laplacian in the (x₁, x₂) plane. It is sparse, with about 5 nonzeros per row.
- `W_grid`, `V_H`: the Hartree potential v_H(x) = ∫ n(x′) w(x − x′) dx′ on the grid, `h * (W @ n)` with W_ij = w(xᵢ − xⱼ).
- `one_electron(V, h, N)`: the lowest eigenpair of T + diag(V), normalized so that h Σ φ² = 1.
- `richardson(E_h, E_2h)`: Richardson extrapolation E₀ ≈ E(h) + (E(h) − E(2h))/3 for an O(h²) method.
- `convergence_order(E_4h, E_2h, E_h)`: the observed order of convergence p = log₂[(E₂ₕ − E₄ₕ)/(Eₕ − E₂ₕ)].

### `exact/`: `exact_H_minus.py`, `exact_He.py`, `exact_H2.py`

1. Build the sparse two-electron Hamiltonian on the N × N grid: N² unknowns, about 40,000 for N = 201.
2. Find the lowest eigenpair with the sparse Lanczos solver `scipy.sparse.linalg.eigsh`.
3. Reshape the eigenvector to Ψ(xᵢ, xⱼ), fix its sign, and normalize it on the grid.
4. Density: n(xᵢ) = 2 h Σⱼ Ψᵢⱼ².
5. `exact_H2.py` adds the nuclear repulsion w(R) to get the total energy.

### `KS/`: `KS_H_minus.py`, `KS_He.py`, `KS_H2.py`: the self-consistent loop

1. **Initial guess:** the lowest orbital with no electron–electron repulsion, n = 2φ².
2. **Hartree potential:** v_H = h W n. It is built from both electrons, so it includes each electron repelling itself.
3. **Exchange:** both electrons share one orbital, so exactly half of v_H is self-repulsion. Exact exchange removes it: v_x = −v_H/2, v_c = 0.
4. **Solve** [T + v_ext + v_H + v_x] φ = ε φ for the lowest orbital, and set n_new = 2φ².
5. **Mix** n ← α n_new + (1 − α) n (default α = 0.4) and repeat until |n_new − n| < 10⁻⁶. The loop is capped at `max_iter`.
6. **Energy:** E = T_s + E_ext + E_H + E_x (+ E_nn for H₂), where
   - T_s = 2⟨φ|−½∂²|φ⟩
   - E_ext = ∫ n v_ext
   - E_H = ½∫ n v_H
   - E_x = −E_H/2
7. **Check:** E must also equal 2ε − E_H/2 (+ E_nn). Adding the two orbital energies counts the electron–electron repulsion twice, and this removes one copy. The two routes agree to about 10⁻⁷ Ha.

`KS_H2.py` imposes left–right symmetry on the density at every step. Without it, the charge sloshes from one nucleus to the other at large R and the loop never converges.

For two electrons in a singlet, Kohn–Sham with exact exchange and no correlation **is** restricted Hartree–Fock. So E_exact − E_KS is the exact correlation energy E_c.

### `compare.py`: outputs

`compare.py` saves the plots below in `results/` and prints every number quoted in the report to the terminal: energies, energy terms and the two energy routes, the HOMO test, density peaks, the H₂ energy curve (with E_nn and the electronic energy), the pair probability, grid convergence with Richardson extrapolation, box-size convergence, SCF mixing and the charge-sloshing test.

| Plot | Content |
|---|---|
| `results/energy_errors.png` | E_method − E_exact (mHa) for H⁻, He and H₂ (R = 3.5). For KS this is −E_c. |
| `results/H2_energy_curve.png` | H₂ energy curve E(R) for R = 1 to 10 with the exact R → ∞ limit, and its error E_method − E_exact against R |
| `results/densities.png` | Top row: n(x) for every method. Bottom row: n_method(x) − n_exact(x). One column each for H⁻, He and H₂ (R = 3.5). |

---

## 4. Analyses and results

All numbers are from `compare.py` at h = 0.1, with box half-width L = 10 for the atoms.

### 4.1 Energies (exact vs Kohn–Sham with exact exchange)

| System | E exact | E KS (= RHF) | E_c = E_exact − E_KS | I exact | −ε KS |
|---|---|---|---|---|---|
| H⁻ | −0.73043 | −0.69182 | −0.03861 | 0.0606 | 0.0523 |
| He | −2.23855 | −2.22450 | −0.01405 | 0.7549 | 0.7503 |
| H₂ (R = 3.5) | −1.36357 | −1.26423 | −0.09933 | – | 0.5046 |

- E_KS > E_exact in every case, as the variational principle requires (same grid and same kinetic matrix in both codes).
- I exact = E(one-electron system) − E(two-electron system) is the true ionization energy. In exact Kohn–Sham, −ε of the highest orbital equals I; Hartree–Fock's −ε (Koopmans' theorem) gets close.
- Correlation is almost three times larger in H⁻ than in He: the loosely bound electrons have more room to avoid each other.

### 4.2 H₂: where mean field breaks

| R | Exact | KS (= RHF) | E_c |
|---|---|---|---|
| 1.0 | −1.41320 | −1.39452 | −0.019 |
| 1.5 | −1.45172 | −1.42639 | −0.025 |
| 2.0 | −1.44200 | −1.40591 | −0.036 |
| 3.5 | −1.36357 | −1.26423 | −0.099 |
| 6.0 | −1.33930 | −1.12054 | −0.219 |
| 10.0 | −1.33963 | −1.05706 | −0.283 |

- The exact curve has its minimum between R ≈ 1.5 and 2 (bond length) and tends to 2 × E(H atom) = −1.3397 Ha as R → ∞.
- RHF ends about 0.28 Ha too high at R = 10 and is still rising. This is **static correlation**. The restricted wavefunction φ(x₁)φ(x₂) puts the two electrons on the same atom 50% of the time at any R, so half of it is "H⁻ + H⁺" even when the atoms are far apart.
- A direct check: at R = 3.5 the probability that both electrons are on the same side of the molecule is **0.145** in the exact wavefunction and **0.485** in RHF.
- Unrestricted HF (separate orbitals for each spin) would fix the energy at large R, at the price of breaking the spin symmetry.

### 4.3 Numerical checks

`compare.py` prints all of these (sections 3–5 of its output).

- **Grid convergence (He):** the error scales as h². Energies at h = 0.2, 0.1 and 0.05 extrapolate to −2.238258 (exact) and −2.224209 (KS). At h = 0.1 the error is about 0.3 mHa, far below every physical effect we discuss.
- **Box size:** a box of L = 10 is converged to better than 1 mHa for He and H⁻ in KS.
- **SCF mixing:** iterations fall from about 110 at α = 0.1 to 6–8 at α ≈ 0.8–1.0 for He. For H₂ the optimum is near α ≈ 0.7–0.8.
- **Charge sloshing:** for H₂ at R = 6 the unsymmetrized loop never converges (the error stays near 2). Imposing the mirror symmetry of the density, which the exact ground state has, makes it converge in 22 iterations.

---

## 5. LDA: discussed, not implemented

Exact exchange is only "local" here because there are two electrons (section 6.2 explains why). Real DFT codes instead approximate the whole exchange–correlation energy, and the simplest approximation is the **local density approximation (LDA)**. We decided not to implement it, but it is the natural next step, so this section sets out what it is and what it would show.

### 5.1 What LDA does

There is one system where the exchange energy can be worked out on paper: the **uniform electron gas**, where the density is the same everywhere. LDA treats each point of a real, lumpy system as a small piece of uniform gas with the local density n(x), and adds up the pieces. For the 1D soft-Coulomb interaction, putting the uniform-gas density matrix into the exchange integral gives (problem statement, Phase 4):

```
n e_x(n) = -(n²/2) J(πn/2),   J(k) = ∫₀^∞ sin²t / (t² sqrt(t² + k²)) dt
E_x      = ∫ n e_x(n(x)) dx
v_x(x)   = d[n e_x(n)]/dn  evaluated at n = n(x)
```

J(k) has no closed form, so it would be tabulated once on a log-spaced grid in n and splined in ln n. In the self-consistent loop, the only change from `KS/` would be the exchange line: v_x = v_x(n(x)) instead of −v_H/2, together with the full v_H.

- **Why "local":** the potential at x depends only on the density at x. It only adds to the diagonal of the KS Hamiltonian, so the matrix stays tridiagonal (sparse in 3D), and the table is reused for every system and iteration.
- **Why approximate:** a real atom is not a uniform gas. In particular, LDA does not cancel each electron's repulsion of itself exactly, as −v_H/2 does for two electrons. What is left over is the **self-interaction error**.
- **Analogy:** estimating a hilly country's farmland value by treating every small patch as flat land at that patch's altitude. It works where the land changes slowly, and badly near cliffs.

### 5.2 What LDA would show here

Reference values for LDA exchange only (no LDA correlation), at h = 0.1 and L = 10. The He energy and exchange energy are also given in the problem statement. The H₂ and H⁻ values come from a reference run made while planning the project; **they are not produced by code in this repository**.

| System | E exact | E KS (exact exchange) | E KS-LDA exchange | −ε LDA |
|---|---|---|---|---|
| He | −2.23855 | −2.22450 | −2.158 | 0.467 |
| H₂ (R = 3.5) | −1.36357 | −1.26423 | −1.238 | 0.351 |
| H⁻ | −0.73043 | −0.69182 | −0.644 (box-dependent) | −0.129 (**unbound**) |

- **Exchange error dominates in He.** LDA gives about 9% too little exchange (about −0.654 against −0.724 Ha). The total LDA error relative to exact exchange is about 0.066 Ha, roughly five times the entire correlation energy (0.014 Ha). So in 1D He, choosing an approximate exchange costs far more than ignoring correlation.
- **The HOMO is too shallow.** The LDA potential dies off exponentially instead of like −1/|x|, so −ε ≈ 0.467 against the true ionization energy I = 0.755 for He. Exact exchange gets 0.750.
- **LDA does not bind H⁻.** The highest orbital energy is positive, so the second electron is held only by the walls of the box, and its energy changes with the box size. This is a known failure of LDA for anions, caused by self-interaction.
- **No local functional fixes static correlation.** For stretched H₂, LDA lies a little below RHF at large R but still far above the exact curve (about 0.22 Ha too high at R = 10). The exact exchange–correlation potential builds a peak at the bond midpoint that holds the electrons on separate atoms; a local functional cannot build it.
- **Correlation:** the LDA correlation fit for this exact interaction (Helbig et al., 2011) would recover only about 0.014 Ha in He, much less than the exchange error, so it would not change these conclusions.

---

## 6. Discussion: why two electrons are trivial, and what the code reveals about cost

### 6.1 For two electrons, none of this is computationally hard

The exact solve takes seconds: 40,000 unknowns for N_g = 201 grid points. With two electrons, Kohn–Sham has no real speed advantage, and LDA is no cheaper than exact exchange, since both only add to the diagonal of a tridiagonal matrix. For N = 2 the methods are compared on **accuracy**: the exact solution is the reference, and Kohn–Sham is the test.

The case for Kohn–Sham is about how cost **grows**, and that shows up in the data structures the code builds.

### 6.2 What the implementation reveals

**The exact solver stores a function of every electron's position at once.**
- Ψ(x₁, x₂) has N_g² entries. For N electrons in d dimensions it has (N_g^d)^N entries, because the grid is *multiplied* once per electron.
- The Hamiltonian built in `H_two_electron` is (N_g^d)^N × (N_g^d)^N. It is sparse, with about 2dN + 1 nonzeros per row, because each electron moves to its neighbours along each axis.
- The Lanczos solver needs sparse matrix–vector products and a few dozen stored vectors of that length.
- The term that forces this joint grid is w(x₁ − x₂), which depends on both coordinates at once.
- For N ≥ 3 the lowest eigenvector of this Hamiltonian is no longer automatically a valid fermion state: it would also have to be projected onto the right permutation symmetry.

| Two electrons, N_g = 50 per axis | Exact unknowns | Memory per vector | KS unknowns per orbital |
|---|---|---|---|
| 1D | 2,500 | 20 KB | 50 |
| 2D | 6.3 × 10⁶ | 50 MB | 2,500 |
| 3D | 1.6 × 10¹⁰ | 125 GB | 125,000 |

**Kohn–Sham stores N functions of one position each.**
- Each orbital has N_g^d entries, so the grid is *added* once per orbital.
- The coupling w(x₁ − x₂) is replaced by v_H(x), the repulsion averaged over where the other electrons usually are. Once the other electrons are averaged out, every equation depends on one coordinate only.
- In our 1D code:
  - the KS Hamiltonian is tridiagonal, N_g × N_g;
  - `eigh_tridiagonal` with `select` returns only the occupied orbitals, at a cost of about N_g per orbital;
  - the SCF loop needs about 20–30 iterations, almost independent of the grid.
- **On larger grids the most expensive step per iteration is the Hartree potential, not the eigensolve.** The dense W matrix costs N_g² memory and an N_g² matrix–vector product, and `V_H` rebuilds W every call. In a quick timing at N_g = 801, one `V_H` call took about 2.9 ms against about 0.45 ms for the eigensolve; at N_g = 201 the two were about equal. Building W once, or computing the convolution with an FFT (N_g log N_g), would remove most of that cost. This is an implementation choice, not a feature of Kohn–Sham.
- What doesn't happen matters too: v_H depends only on the density, so its cost doesn't grow with the number of electrons.

**What exchange costs.**
- For two electrons, exact exchange is the local term −v_H/2. That is a coincidence of N = 2: there is only one orbital, so the exchange integral collapses back onto it.
- For N > 2, Hartree–Fock exchange is an integral operator, (K_j φ_i)(x) = φ_j(x) ∫ φ_j(x′) φ_i(x′) w(x − x′) dx′. Its value at x depends on φ_i everywhere, and it acts differently on each orbital. On the grid that makes the Fock matrix **dense** (N_g × N_g). Building it costs about (N/2)² N_g² per iteration, and diagonalizing it costs about N_g³ instead of about N_g.
- LDA exchange is a **local** function v_x(n(x)). It only adds to the diagonal, so the matrix stays tridiagonal (sparse in 3D). The J(k) table is built once and reused for every system and iteration, so each iteration costs about N_g spline evaluations.

### 6.3 Why Kohn–Sham with LDA is the method of choice for many electrons

| Method | What it stores | Formal scaling with system size |
|---|---|---|
| Exact (full CI, or this grid Ψ) | Ψ of all coordinates | exponential |
| Coupled cluster CCSD(T) | Correlation built on HF | ~N⁷ |
| Hartree–Fock | N orbitals + nonlocal exchange | ~N⁴ |
| Kohn–Sham with LDA/GGA | N orbitals + local v_xc | ~N³ |

These are textbook formal scalings; production codes reduce both HF and DFT further with screening and density fitting.

- **Most of the saving comes from going to orbitals at all.** Mean field removes the exponential wall, and Hartree–Fock already gets that.
- **Kohn–Sham with a local functional adds the rest.** It avoids HF's nonlocal exchange, keeps the Hamiltonian sparse, and includes an approximate correlation energy that Hartree–Fock lacks entirely.
- **The price is accuracy.** Our two-electron results measure it exactly:
  - small errors when the electrons are tightly bound (He: 14 mHa of missing correlation, measured here; about 66 mHa of LDA exchange error, from the reference values in section 5.2);
  - qualitative failures where self-interaction matters (LDA leaves H⁻ unbound, section 5.2);
  - failure where electrons need to avoid each other strongly (stretched H₂: about 0.28 Ha with RHF, measured here; about 0.22 Ha with LDA, section 5.2).

The fair summary is "approximate functionals fail", not "DFT fails": Kohn–Sham DFT with the exact functional would reproduce every exact number here.

### 6.4 Planned: measured time and space complexity

To be added later. The plan:

1. **Grid scaling, two electrons:** wall time (eigensolve only, median of several runs) and peak memory of the exact solver and the KS solver for N_g = 50–400 in 1D, plus a coarse 2D exact solve (N_g ≈ 10–40 per axis). Fit log–log slopes with uncertainties. The expected memory slopes are about 2 in N_g for 1D exact and about 4 for 2D exact.
2. **Many-electron Kohn–Sham:** occupy N/2 orbitals for N = 2 to 32 electrons in a 1D potential, time each SCF run against N, and compare with the analytic exact cost (N_g^N). Beyond two electrons this needs a local functional such as LDA (section 5), so it depends on implementing it.
3. **Hartree potential:** compare the dense W matrix with FFT convolution, to separate how the code is written from the intrinsic cost of the method.

---

## 7. Limitations

- The model is 1D with softened interactions. Absolute numbers don't transfer to real 3D atoms, but the qualitative failures (self-interaction, the shallow HOMO, static correlation) are the same ones seen in real DFT.
- LDA is not implemented. The LDA numbers in section 5.2 are reference values, not output of this code. LDA correlation (the QMC-based fit of Helbig et al., 2011) was never part of the plan.
- Only spin-restricted, closed-shell calculations are done. Unrestricted calculations for stretched H₂ are not included.
- All results use finite boxes. This matters only for systems that are barely bound or unbound, such as H⁻ (and would matter a lot for H⁻ with LDA, which leaves it unbound).

## 8. Corrections to earlier reference values

The reference table in the early notebook listed RHF energies for stretched H₂ at R ≥ 4 (−1.1904, −1.0693 and −1.0224 Ha at R = 4, 6 and 10), plus an LDA-X column. Those came from an unconverged, sloshing SCF loop. The correct converged values are in section 4.2. The exact-solver column of that table was right.

## 9. References

- P. Hohenberg and W. Kohn, Phys. Rev. 136, B864 (1964).
- W. Kohn and L. J. Sham, Phys. Rev. 140, A1133 (1965).
- K. Capelle, *A Bird's-Eye View of Density-Functional Theory*, arXiv:cond-mat/0211443.
- N. Helbig et al., Phys. Rev. A 83, 032503 (2011): 1D soft-Coulomb LDA.
- L. O. Wagner, E. M. Stoudenmire, K. Burke and S. R. White, Phys. Chem. Chem. Phys. 14, 8581 (2012): reference calculations in one dimension.
- A. J. Cohen, P. Mori-Sánchez and W. Yang, Science 321, 792 (2008): delocalization and static correlation error.
- A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry*, ch. 3 (Hartree–Fock).
