# Exact vs Kohn–Sham DFT for a 1D two-electron system - EP4210 Computational Physics project

### - By Siddharth Shivakumar (EP25BTECH11027), Kushal R (EP25BTECH11012), and Zulfqar Farooq (EP25BTECH11036)
We solve a two-electron problem **exactly** on a grid, then solve the same problem with **Kohn–Sham density functional theory (DFT)**, and measure where and why the approximate method goes wrong. Because the exact answer is computable here, every error in the approximate method can be measured directly instead of guessed.

The Kohn–Sham calculations use exact exchange, which for two electrons is the same as restricted Hartree–Fock. The local density approximation (LDA), the functional real DFT codes are built on, is discussed in section 5 but not implemented.

All quantities are in Hartree atomic units ($\hbar = m_e = e = 1$): energies in Hartree (Ha, 1 Ha = 27.211 eV), lengths in bohr.

---

## Problem statement

**In one line.** Benchmark restricted Hartree–Fock against exact solutions of the one-dimensional soft-Coulomb H⁻, He and H₂, and measure the correlation energy, including how it grows as H₂ is stretched.

**Goal.** Solve a two-electron problem on a line exactly. Then solve the same problem with restricted Hartree–Fock, which for two electrons is the same as Kohn–Sham with exact exchange, and compare the energies and densities of the two methods. The difference between the two energies is exactly the correlation energy, so the error of the mean-field method can be measured directly.

**Model.** Two electrons on a line, with every Coulomb interaction softened to $w(x) = 1/\sqrt{x^2 + 1}$. Section 1 gives the Hamiltonian and the three systems: H⁻, He and H₂ (whose total energy includes the nuclear repulsion $w(R)$).

**Tasks.**

1. **Grid tools.** A uniform grid on $[-L, L]$ with spacing $h$ and an odd number of points, the three-point finite-difference kinetic matrix, and a one-electron solver for the lowest eigenpair of $T + \mathrm{diag}(V)$.
2. **Exact solver.** The sparse Hamiltonian $T \otimes I + I \otimes T + \mathrm{diag}(V)$ on the $(x_1, x_2)$ grid, its lowest eigenpair by the Lanczos method, and the density $n(x)$ from $\Psi(x_1, x_2)$. For H⁻, He and H₂.
3. **Hartree–Fock solver.** A self-consistent loop for one doubly occupied orbital: start from the bare-nucleus orbital, build $v_\text{eff} = v_\text{ext} + \tfrac{1}{2} v_H$, mix the density with $\alpha = 0.4$, and stop when the change in the density falls below $10^{-6}$. For H₂, keep the density symmetric about $x = 0$.
4. **Energy.** $E = T_s + E_\text{ext} + E_H + E_x$ term by term (plus $E_{nn} = w(R)$ for H₂), checked against the eigenvalue route $2\varepsilon - \tfrac{1}{2} E_H$ (plus $E_{nn}$).
5. **Comparison and checks** (`compare.py`). Exact against Hartree–Fock energies and densities for H⁻, He and H₂ at $R = 3.5$; the H₂ energy curve from $R = 1$ to $10$, and how often both electrons sit on the same side of the molecule; convergence in the grid spacing (with Richardson extrapolation) and in the box size; and the behaviour of the loop for stretched H₂.

**Phases.** The original problem statement (`PS.md`) divides the work into six phases. This project carries out Phases 1–3. Phases 4–6 are future extensions that we have not started.

| Phase | Content | Status |
|---|---|---|
| 1 | One-electron solver, three-point kinetic energy, Richardson extrapolation | Done |
| 2 | Exact two-electron solver | Done |
| 3 | Kohn–Sham with exact exchange (= restricted Hartree–Fock), giving the exact correlation energy | Done |
| 4 | LDA exchange for the 1D soft-Coulomb electron gas | Future extension |
| 5 | LDA correlation from a published parametrisation | Future extension |
| 6 | Exact exchange–correlation potential, by inverting the exact density | Future extension |

Besides the phases, the problem statement suggests stretched H₂ as an extension and asks for a grid-convergence study; both are part of this work. Its other suggested studies, such as scans over the nuclear charge $Z$ and the softening parameter, were not done.

---

## 1. The model

Two electrons on a line. Every Coulomb interaction is softened so nothing diverges:

```math
w(x) = \frac{1}{\sqrt{x^2 + a^2}}, \qquad a = 1
```

```math
\hat{H} = -\frac{1}{2}\frac{\partial^2}{\partial x_1^2} - \frac{1}{2}\frac{\partial^2}{\partial x_2^2} + v_\text{ext}(x_1) + v_\text{ext}(x_2) + w(x_1 - x_2)
```

| System | External potential | Notes |
|---|---|---|
| H⁻ | $v_\text{ext} = -w(x)$, one nucleus with $Z = 1$ at $x = 0$ | Weakly bound: the second electron is held by only about 0.06 Ha |
| He | $v_\text{ext} = -2w(x)$, $Z = 2$ | Tightly bound; the cleanest benchmark |
| H₂ | $v_\text{ext} = -w(x - R/2) - w(x + R/2)$, two nuclei | Total energy also includes the nucleus–nucleus repulsion $w(R)$ |

- **Why soften?** The bare 1D Coulomb potential $1/\lvert x \rvert$ has no finite ground state: the electron collapses onto the nucleus. With $a = 1$, $w(0) = 1$ and a plain grid converges. The same $a$ is used for every interaction, so every method describes the same physical system.
- **Spin.** $\hat{H}$ has no spin operators, so the two-electron ground state is a nodeless, symmetric spatial $\Psi(x_1, x_2)$ paired with the spin singlet.
- **Density.** $n(x) = 2 \int \lvert \Psi(x, x') \rvert^2 \, dx'$, with $\int n \, dx = 2$.
- **Born–Oppenheimer.** For H₂ the nuclei are held fixed. Their repulsion $w(R)$ is a constant at fixed $R$. It doesn't change the wavefunction, but it is part of the total energy at that geometry, and it creates the minimum in the energy curve $E(R)$.

---

## 2. How to run

You need **Python 3.9 or newer**. The code uses three packages, NumPy, SciPy and Matplotlib, which are listed in `requirements.txt`. If `python` is not found on your computer, use `python3` (macOS and Linux) or `py` (Windows) in the commands below.

**Step 1. Get the code.** Clone the repository and move into its folder:

```bash
git clone https://github.com/PiCamel/Exact-Against-HF-1D-Soft-Coulomb-Solver.git
```

```bash
cd Exact-Against-HF-1D-Soft-Coulomb-Solver
```

You can also download the ZIP from the GitHub page, unzip it and open a terminal in that folder.

**Step 2 (optional). Make a virtual environment**, so the packages stay separate from the rest of your system:

```bash
python -m venv .venv
```

Then activate it with `source .venv/bin/activate` on macOS or Linux, or `.venv\Scripts\activate` on Windows. (On Debian or Ubuntu, `venv` may need installing first with `sudo apt install python3-venv`.)

**Step 3. Install the packages:**

```bash
pip install -r requirements.txt
```

**Step 4. Run the full comparison** from the project folder:

```bash
python compare.py
```

This runs every solver, prints every number used in the report, and saves three plots in `results/`. It takes one to two minutes on a laptop; most of that time is the exact solver on the finest grids.

**Step 5 (optional). Run a single solver.** Every file in `exact/` and `KS/` also runs on its own. It prints its energies and opens a plot of the density; close the plot window to finish.

```bash
python KS/KS_He.py
```

The other solvers run the same way: `python exact/exact_He.py`, `python exact/exact_H2.py`, `python KS/KS_H2.py`, and so on.

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
report/            the project report (report.tex and report.pdf)
requirements.txt   Python packages to install (pip install -r requirements.txt)
```

Each method has one folder, and each folder has one file per system: H⁻ ($Z = 1$), He ($Z = 2$) and H₂.

| File | What it does |
|---|---|
| `common.py` | Shared building blocks: the grid, the soft-Coulomb interaction `w`, external potentials, the kinetic-energy matrix, the two-electron Hamiltonian, the Hartree potential and the one-electron solver. |
| `exact/exact_H_minus.py` | Exact two-electron ground state for one nucleus of charge $Z$ (default $Z = 1$: H⁻). |
| `exact/exact_He.py` | The same for He ($Z = 2$). |
| `exact/exact_H2.py` | Exact two-electron ground state for H₂ at internuclear distance $R$. |
| `KS/KS_H_minus.py` | Kohn–Sham with exact exchange for one nucleus (default $Z = 1$: H⁻). For two electrons this is restricted Hartree–Fock (RHF). |
| `KS/KS_He.py` | The same for He ($Z = 2$). |
| `KS/KS_H2.py` | Kohn–Sham with exact exchange (= RHF) for H₂. |
| `compare.py` | Runs every solver and saves the energy and density comparison plots in `results/`. |
| `requirements.txt` | The Python packages the code needs: NumPy, SciPy and Matplotlib. |

Every analysis file has a `run(...)` function that returns a dictionary of results (energies, density, grid, convergence history), so `compare.py` can use it, plus a `__main__` block for running it alone. Each solver file adds the project folder to `sys.path` at the top, so it can import `common.py` whether you run it as `python KS/KS_He.py` from the project folder or from inside its own folder.

### `common.py`

- `make_grid(L, h)`: uniform grid from $-L$ to $L$ with spacing $h$ and an odd number of points, so $x = 0$ is on the grid.
- `w`, `v_1atom`, `v_2atom`, `v_ext`, `v_ext_H2`: interaction and external potentials.
- `kinetic_matrix(N, h)` / `T_1e(N, h)`: $-\tfrac{1}{2}\, d^2/dx^2$ as a tridiagonal matrix, from the 3-point finite difference $\varphi''_i \approx (\varphi_{i+1} - 2\varphi_i + \varphi_{i-1})/h^2$. The error is $O(h^2)$. Wavefunctions are set to zero outside the box.
- `H_two_electron(x, h, v)`: the two-electron Hamiltonian on the $N \times N$ grid, built as `kron(T, I) + kron(I, T) + diag(V)`. Each electron's kinetic energy acts along its own index, so this is the 5-point Laplacian in the $(x_1, x_2)$ plane. It is sparse, with about 5 nonzeros per row.
- `W_grid`, `V_H`: the Hartree potential $v_H(x) = \int n(x')\, w(x - x')\, dx'$ on the grid, `h * (W @ n)` with $W_{ij} = w(x_i - x_j)$.
- `one_electron(V, h, N)`: the lowest eigenpair of $T + \mathrm{diag}(V)$, normalized so that $h \sum_i \varphi_i^2 = 1$.
- `richardson(E_h, E_2h)`: Richardson extrapolation $E_0 \approx E(h) + [E(h) - E(2h)]/3$ for an $O(h^2)$ method.
- `convergence_order(E_4h, E_2h, E_h)`: the observed order of convergence $p = \log_2\left[(E_{2h} - E_{4h})/(E_h - E_{2h})\right]$.

### `exact/`: `exact_H_minus.py`, `exact_He.py`, `exact_H2.py`

1. Build the sparse two-electron Hamiltonian on the $N \times N$ grid: $N^2$ unknowns, about 40,000 for $N = 201$.
2. Find the lowest eigenpair with the sparse Lanczos solver `scipy.sparse.linalg.eigsh`.
3. Reshape the eigenvector to $\Psi(x_i, x_j)$, fix its sign, and normalize it on the grid.
4. Density: $n(x_i) = 2h \sum_j \Psi_{ij}^2$.
5. `exact_H2.py` adds the nuclear repulsion $w(R)$ to get the total energy.

### `KS/`: `KS_H_minus.py`, `KS_He.py`, `KS_H2.py`: the self-consistent loop

1. **Initial guess:** the lowest orbital with no electron–electron repulsion, $n = 2\varphi^2$.
2. **Hartree potential:** $v_H = h W n$. It is built from both electrons, so it includes each electron repelling itself.
3. **Exchange:** both electrons share one orbital, so exactly half of $v_H$ is self-repulsion. Exact exchange removes it: $v_x = -v_H/2$, $v_c = 0$.
4. **Solve** $[T + v_\text{ext} + v_H + v_x]\,\varphi = \varepsilon \varphi$ for the lowest orbital, and set $n_\text{new} = 2\varphi^2$.
5. **Mix** $n \leftarrow \alpha\, n_\text{new} + (1 - \alpha)\, n$ with $\alpha = 0.4$ (fixed in every KS file) and repeat until $\lVert n_\text{new} - n \rVert < 10^{-6}$. The loop is capped at `max_iter`.
6. **Energy:** $E = T_s + E_\text{ext} + E_H + E_x$ ($+\,E_{nn}$ for H₂), where
   - $T_s = 2\langle \varphi \rvert -\tfrac{1}{2}\partial_x^2 \lvert \varphi \rangle$
   - $E_\text{ext} = \int n\, v_\text{ext}\, dx$
   - $E_H = \tfrac{1}{2}\int n\, v_H\, dx$
   - $E_x = -E_H/2$
7. **Check:** $E$ must also equal $2\varepsilon - E_H/2$ ($+\,E_{nn}$). Adding the two orbital energies counts the electron–electron repulsion twice, and this removes one copy. The two routes agree to about $10^{-7}$ Ha.

`KS_H2.py` imposes left–right symmetry on the density at every step. Without it, the charge sloshes from one nucleus to the other at large $R$ and the loop never converges.

For two electrons in a singlet, Kohn–Sham with exact exchange and no correlation **is** restricted Hartree–Fock. So $E_\text{exact} - E_\text{KS}$ is the exact correlation energy $E_c$.

### `compare.py`: outputs

`compare.py` saves the plots below in `results/` and prints every number quoted in the report to the terminal: energies, energy terms and the two energy routes, density peaks, the H₂ energy curve (with $E_{nn}$ and the electronic energy), the pair probability, grid convergence with Richardson extrapolation, box-size convergence and the charge-sloshing test.

| Plot | Content |
|---|---|
| `results/energy_errors.png` | $E_\text{method} - E_\text{exact}$ (mHa) for H⁻, He and H₂ ($R = 3.5$). For KS this is $-E_c$. |
| `results/H2_energy_curve.png` | H₂ energy curve $E(R)$ for $R = 1$ to $10$ with the exact $R \to \infty$ limit, and its error $E_\text{method} - E_\text{exact}$ against $R$ |
| `results/densities.png` | Top row: $n(x)$ for every method. Bottom row: $n_\text{method}(x) - n_\text{exact}(x)$. One column each for H⁻, He and H₂ ($R = 3.5$). |

---

## 4. Analyses and results

All numbers are from `compare.py` at $h = 0.1$, with box half-width $L = 10$ for the atoms.

### 4.1 Energies (exact vs Kohn–Sham with exact exchange)

| System | $E_\text{exact}$ | $E_\text{KS}$ (= RHF) | $E_c = E_\text{exact} - E_\text{KS}$ |
|---|---|---|---|
| H⁻ | −0.73043 | −0.69182 | −0.03861 |
| He | −2.23855 | −2.22450 | −0.01405 |
| H₂ ($R = 3.5$) | −1.36357 | −1.26423 | −0.09933 |

- $E_\text{KS} > E_\text{exact}$ in every case, as the variational principle requires (same grid and same kinetic matrix in both codes).
- Correlation is almost three times larger in H⁻ than in He: the loosely bound electrons have more room to avoid each other.

### 4.2 H₂: where mean field breaks

| $R$ | Exact | KS (= RHF) | $E_c$ |
|---|---|---|---|
| 1.0 | −1.41320 | −1.39452 | −0.019 |
| 1.5 | −1.45172 | −1.42639 | −0.025 |
| 2.0 | −1.44200 | −1.40591 | −0.036 |
| 3.5 | −1.36357 | −1.26423 | −0.099 |
| 6.0 | −1.33930 | −1.12054 | −0.219 |
| 10.0 | −1.33963 | −1.05706 | −0.283 |

- The exact curve has its minimum between $R \approx 1.5$ and $2$ (bond length) and tends to $2E(\text{H atom}) = -1.3397$ Ha as $R \to \infty$.
- RHF ends about 0.28 Ha too high at $R = 10$ and is still rising. This is **static correlation**. The restricted wavefunction $\varphi(x_1)\varphi(x_2)$ puts the two electrons on the same atom 50% of the time at any $R$, so half of it is "H⁻ + H⁺" even when the atoms are far apart.
- A direct check: at $R = 3.5$ the probability that both electrons are on the same side of the molecule is **0.145** in the exact wavefunction and **0.485** in RHF.
- Unrestricted HF (separate orbitals for each spin) would fix the energy at large $R$, at the price of breaking the spin symmetry.

### 4.3 Numerical checks

`compare.py` prints all of these (sections 3–5 of its output).

- **Grid convergence (He):** the error scales as $h^2$. Energies at $h = 0.2$, $0.1$ and $0.05$ extrapolate to −2.238258 (exact) and −2.224209 (KS). At $h = 0.1$ the error is about 0.3 mHa, far below every physical effect we discuss.
- **Box size:** a box of $L = 10$ is converged to better than 1 mHa for He and H⁻ in KS.
- **Charge sloshing:** for H₂ at $R = 6$ the unsymmetrized loop never converges (the error is still about 2.7 after 300 iterations). Imposing the mirror symmetry of the density, which the exact ground state has, makes it converge in 22 iterations.

---

## 5. Local Density Approximation (LDA):

Exact exchange is only "local" here because there are two electrons (section 6.2 explains why). Real DFT codes instead approximate the whole exchange–correlation energy, and the simplest approximation is the **local density approximation (LDA)**. It is not implemented in this project: it is Phase 4, a future extension. This section sets out what it is and what the problem statement expects it to show.

### 5.1 What LDA does

There is one system where the exchange energy can be worked out on paper: the **uniform electron gas**, where the density is the same everywhere. LDA treats each point of a real, lumpy system as a small piece of uniform gas with the local density $n(x)$, and adds up the pieces. For the 1D soft-Coulomb interaction, putting the uniform-gas density matrix into the exchange integral gives (problem statement, Phase 4):

```math
n\, e_x(n) = -\frac{n^2}{2}\, J\!\left(\frac{\pi n}{2}\right), \qquad J(k) = \int_0^\infty \frac{\sin^2 t}{t^2 \sqrt{t^2 + k^2}}\, dt
```

```math
E_x = \int n\, e_x\big(n(x)\big)\, dx, \qquad v_x(x) = \left.\frac{d\,[n\, e_x(n)]}{dn}\right|_{n = n(x)}
```

$J(k)$ has no closed form, so it would be tabulated once on a log-spaced grid in $n$ and splined in $\ln n$. In the self-consistent loop, the only change from `KS/` would be the exchange line: $v_x = v_x(n(x))$ instead of $-v_H/2$, together with the full $v_H$.

- **Why "local":** the potential at $x$ depends only on the density at $x$. It only adds to the diagonal of the KS Hamiltonian, so the matrix stays tridiagonal (sparse in 3D), and the table is reused for every system and iteration.
- **Why approximate:** a real atom is not a uniform gas. In particular, LDA does not cancel each electron's repulsion of itself exactly, as $-v_H/2$ does for two electrons. What is left over is the **self-interaction error**.
- **Analogy:** estimating a hilly country's farmland value by treating every small patch as flat land at that patch's altitude. It works where the land changes slowly, and badly near cliffs.

### 5.2 What LDA would be expected to show

The problem statement quotes reference values from its own calculation for 1D helium with LDA exchange only: a total energy of −2.158 Ha, and an exchange energy of −0.654 Ha against −0.724 Ha for exact exchange. We have not reproduced these numbers. If they hold:

- **Exchange error would dominate in He.** LDA would give about 10% too little exchange, and its total energy would lie about 0.066 Ha above the problem statement's Hartree–Fock value (−2.2242 Ha). That is several times the entire correlation energy we measure (0.014 Ha).
- **No local functional fixes static correlation.** The problem statement expects LDA to fail for stretched H₂ in the same way as restricted Hartree–Fock. The exact exchange–correlation potential builds a peak at the bond midpoint that holds the electrons on separate atoms; a local functional cannot build it.
- **Correlation.** The exact correlation energy of 1D He is only 0.014 Ha (measured here), much smaller than this exchange shortfall, so adding LDA correlation (Phase 5) is unlikely to change the conclusion.

---

## 6. Discussion: Triviality of two electrons, and computational cost

### 6.1 For two electrons, none of this is computationally hard

The exact solve takes seconds: 40,000 unknowns for $N_g = 201$ grid points. With two electrons, Kohn–Sham has no real speed advantage, and LDA is no cheaper than exact exchange, since both only add to the diagonal of a tridiagonal matrix. For $N = 2$ the methods are compared on **accuracy**: the exact solution is the reference, and Kohn–Sham is the test.

The case for Kohn–Sham is about how cost **grows**, and that shows up in the data structures the code builds.

### 6.2 What the implementation reveals

**The exact solver stores a function of every electron's position at once.**
- $\Psi(x_1, x_2)$ has $N_g^2$ entries. For $N$ electrons in $d$ dimensions it has $(N_g^d)^N$ entries, because the grid is *multiplied* once per electron.
- The Hamiltonian built in `H_two_electron` is $(N_g^d)^N \times (N_g^d)^N$. It is sparse, with about $2dN + 1$ nonzeros per row, because each electron moves to its neighbours along each axis.
- The Lanczos solver needs sparse matrix–vector products and a few dozen stored vectors of that length.
- The term that forces this joint grid is $w(x_1 - x_2)$, which depends on both coordinates at once.
- For $N \ge 3$ the lowest eigenvector of this Hamiltonian is no longer automatically a valid fermion state: it would also have to be projected onto the right permutation symmetry.

| Two electrons, $N_g = 50$ per axis | Exact unknowns | Memory per vector | KS unknowns per orbital |
|---|---|---|---|
| 1D | 2,500 | 20 KB | 50 |
| 2D | $6.3 \times 10^6$ | 50 MB | 2,500 |
| 3D | $1.6 \times 10^{10}$ | 125 GB | 125,000 |

**Kohn–Sham stores $N$ functions of one position each.**
- Each orbital has $N_g^d$ entries, so the grid is *added* once per orbital.
- The coupling $w(x_1 - x_2)$ is replaced by $v_H(x)$, the repulsion averaged over where the other electrons usually are. Once the other electrons are averaged out, every equation depends on one coordinate only.
- In our 1D code:
  - the KS Hamiltonian is tridiagonal, $N_g \times N_g$;
  - `eigh_tridiagonal` with `select` returns only the occupied orbitals, at a cost of about $N_g$ per orbital;
  - the SCF loop needs about 20–30 iterations, almost independent of the grid.
- **On larger grids the most expensive step per iteration is the Hartree potential, not the eigensolve.** The dense $W$ matrix costs $N_g^2$ memory and an $N_g^2$ matrix–vector product, and `V_H` rebuilds $W$ every call. In a quick timing at $N_g = 801$, one `V_H` call took about 2.9 ms against about 0.45 ms for the eigensolve; at $N_g = 201$ the two were about equal. Building $W$ once, or computing the convolution with an FFT ($N_g \log N_g$), would remove most of that cost. This is an implementation choice, not a feature of Kohn–Sham.
- What doesn't happen matters too: $v_H$ depends only on the density, so its cost doesn't grow with the number of electrons.

**What exchange costs.**
- For two electrons, exact exchange is the local term $-v_H/2$. That is a coincidence of $N = 2$: there is only one orbital, so the exchange integral collapses back onto it.
- For $N > 2$, Hartree–Fock exchange is an integral operator, $(K_j \varphi_i)(x) = \varphi_j(x) \int \varphi_j(x')\, \varphi_i(x')\, w(x - x')\, dx'$. Its value at $x$ depends on $\varphi_i$ everywhere, and it acts differently on each orbital. On the grid that makes the Fock matrix **dense** ($N_g \times N_g$). Building it costs about $(N/2)^2 N_g^2$ per iteration, and diagonalizing it costs about $N_g^3$ instead of about $N_g$.
- LDA exchange is a **local** function $v_x(n(x))$. It only adds to the diagonal, so the matrix stays tridiagonal (sparse in 3D). The $J(k)$ table is built once and reused for every system and iteration, so each iteration costs about $N_g$ spline evaluations.

### 6.3 Why Kohn–Sham with LDA is the method of choice for many electrons

| Method | What it stores | Formal scaling with system size |
|---|---|---|
| Exact (full CI, or this grid $\Psi$) | $\Psi$ of all coordinates | exponential |
| Coupled cluster CCSD(T) | Correlation built on HF | $\sim N^7$ |
| Hartree–Fock | $N$ orbitals + nonlocal exchange | $\sim N^4$ |
| Kohn–Sham with LDA/GGA | $N$ orbitals + local $v_{xc}$ | $\sim N^3$ |

These are textbook formal scalings; production codes reduce both HF and DFT further with screening and density fitting.

- **Most of the saving comes from going to orbitals at all.** Mean field removes the exponential wall, and Hartree–Fock already gets that.
- **Kohn–Sham with a local functional adds the rest.** It avoids HF's nonlocal exchange, keeps the Hamiltonian sparse, and includes an approximate correlation energy that Hartree–Fock lacks entirely.
- **The price is accuracy.** Our two-electron results measure it exactly:
  - small errors when the electrons are tightly bound (He: 14 mHa of missing correlation, measured here; about 66 mHa of LDA exchange error if the problem statement's reference values in section 5.2 hold);
  - failure where electrons need to avoid each other strongly (stretched H₂: about 0.28 Ha with RHF, measured here; the problem statement expects LDA to fail in the same way).

The fair summary is that approximate functionals fail: Kohn–Sham DFT with the exact functional would reproduce every exact number here.

### 6.4 Planned: measured time and space complexity

To be added later. The plan:

1. **Grid scaling, two electrons:** wall time (eigensolve only, median of several runs) and peak memory of the exact solver and the KS solver for $N_g = 50$–$400$ in 1D, plus a coarse 2D exact solve ($N_g \approx 10$–$40$ per axis). Fit log–log slopes with uncertainties. The expected memory slopes are about 2 in $N_g$ for 1D exact and about 4 for 2D exact.
2. **Many-electron Kohn–Sham:** occupy $N/2$ orbitals for $N = 2$ to $32$ electrons in a 1D potential, time each SCF run against $N$, and compare with the analytic exact cost ($N_g^N$). Beyond two electrons this needs a local functional such as LDA (section 5), so it depends on implementing it.
3. **Hartree potential:** compare the dense $W$ matrix with FFT convolution, to separate how the code is written from the intrinsic cost of the method.

---

## 7. Limitations

- The model is 1D with softened interactions. Absolute numbers don't transfer to real 3D atoms, but the static-correlation failure of restricted Hartree–Fock in stretched H₂ is the same one seen in real molecules.
- LDA exchange, LDA correlation and the exact exchange–correlation potential (Phases 4–6) are not implemented; they are future extensions. The LDA numbers in section 5.2 are the problem statement's reference values, not output of this code.
- Only spin-restricted, closed-shell calculations are done. Unrestricted calculations for stretched H₂ are not included.
- All results use finite boxes. This matters only for weakly bound systems such as H⁻, whose exact energy still changes in the fourth decimal place between $L = 10$ and $L = 14$.

## 8. References

- P. Hohenberg and W. Kohn, Phys. Rev. 136, B864 (1964).
- W. Kohn and L. J. Sham, Phys. Rev. 140, A1133 (1965).
- K. Capelle, *A Bird's-Eye View of Density-Functional Theory*, arXiv:cond-mat/0211443.
- N. Helbig et al., Phys. Rev. A 83, 032503 (2011): 1D soft-Coulomb LDA.
- L. O. Wagner, E. M. Stoudenmire, K. Burke and S. R. White, Phys. Chem. Chem. Phys. 14, 8581 (2012): reference calculations in one dimension.
- A. J. Cohen, P. Mori-Sánchez and W. Yang, Science 321, 792 (2008): delocalization and static correlation error.
- A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry*, ch. 3 (Hartree–Fock).
