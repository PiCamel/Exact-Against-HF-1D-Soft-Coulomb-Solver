# Project of EP4210 - Computational Physics
Course Project of EP4210 - Computational Physics, by Siddharth, Kushal, and Zulfiqar


# DFT
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

# Problem Statement
## 1D soft-Coulomb two-electron system: exact vs Kohn–Sham DFT

Consider two electrons confined to a line, bound to a nucleus of charge $Z$ at $x=0$. All Coulomb interactions are softened to remove the singularity:

$$
w(x) = \frac{1}{\sqrt{x^2+1}}
$$

$$
\hat H = -\frac12\frac{\partial^2}{\partial x_1^2} - \frac12\frac{\partial^2}{\partial x_2^2} - Z\,w(x_1) - Z\,w(x_2) + w(x_1 - x_2)
$$

The ground state is a spin singlet with a symmetric spatial wavefunction $\Psi(x_1,x_2)$, and the electron density is

$$
n(x) = 2\int |\Psi(x,x')|^2\,dx',\qquad \int n\,dx = 2
$$

Use Hartree atomic units throughout.

## Why computation is necessary

Even the one-electron soft-Coulomb problem has no closed-form solution, and the two-electron problem is a two-dimensional PDE with no analytic solution. It is small enough to solve essentially exactly on a grid. That is what makes it a useful benchmark: for $N$ electrons the grid cost grows as $N_x^N$, which is why approximate methods such as DFT are needed for real systems. Because a numerically exact answer is available here, approximations can be tested against truth, which is not possible for real 3D atoms beyond helium.

## Objective

Determine how accurate approximate Kohn–Sham DFT is for this system, and why it fails where it does. Specifically:

1. Obtain the numerically exact ground-state energy and density of the two-electron system (for example the helium-like case $Z=2$).
2. Treat the same system with Kohn–Sham DFT using approximate exchange–correlation, and compare with the exact result for the total energy, the density, and the ionization energy.
3. Quantify the error and identify which part of the exchange–correlation energy is responsible for it.

## Extension: stretched H$_2$

Replace the single nucleus by two soft nuclei separated by a distance $R$, each with charge $Z=1$, and study the system as $R$ increases. Compare the exact energy curve $E(R)$ with the Kohn–Sham result, in particular the dissociation limit, and explain why the restricted mean-field description breaks down at large $R$.