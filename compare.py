# Compares the exact and Kohn-Sham (exact exchange = RHF) solvers on H-, He and H2.
# Two comparisons, saved as plots in results/:
#   1. Energies:  E_method - E_exact for each system, and the H2 energy curve E(R)
#   2. Densities: n(x) for each method, and n_method(x) - n_exact(x)
# Every method uses the same grid for a given system, so the results can be compared point by point.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg") # write files, no windows
import matplotlib.pyplot as plt

from exact import exact_H_minus, exact_He, exact_H2
from KS import KS_H_minus, KS_He, KS_H2
from common import make_grid, v_ext, one_electron

METHODS = ["exact", "KS"]
APPROX = METHODS[1:] # the approximate (mean-field) methods

SOLVERS = {"H-": {"exact": exact_H_minus, "KS": KS_H_minus},
           "He": {"exact": exact_He,      "KS": KS_He},
           "H2": {"exact": exact_H2,      "KS": KS_H2}}

H = 0.1                 # grid spacing
L = 10.0                # box half-width
R_MAIN = 3.5            # H2 bond length for the single-geometry comparison
R_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

TITLE = {"H-": "H$^-$", "He": "He", "H2": f"H$_2$ (R = {R_MAIN})"}
COL = {"exact": "#1f1f1f", "KS": "#534AB7"}
LAB = {"exact": "exact", "KS": "KS exact exchange (= RHF)"}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("saved", os.path.join("results", name))

# Run every method on every system

res = {}
for name, solver in SOLVERS.items():
    extra = {"R": R_MAIN} if name == "H2" else {}
    res[name] = {m: solver[m].run(L=L, h=H, **extra) for m in METHODS}

print("\nTotal energies (Ha)")
print(f"{'system':8s}" + "".join(f"{m:>12s}" for m in METHODS) + "".join(f"{m + ' - exact':>14s}" for m in APPROX))
for name in SOLVERS:
    E = {m: res[name][m]["E"] for m in METHODS}
    print(f"{name:8s}" + "".join(f"{E[m]:12.5f}" for m in METHODS) + "".join(f"{E[m] - E['exact']:14.5f}" for m in APPROX))
print("KS - exact is minus the exact correlation energy (E_c = E_exact - E_KS).")

# H2 energy curve: the box grows with R so the density is not squashed by the walls
print("\nH2 energy curve")
sweep = {m: [] for m in METHODS}
for R in R_SWEEP:
    for m in METHODS:
        sweep[m].append(SOLVERS["H2"][m].run(R=R, L=max(L, R/2 + 9), h=H)["E"])
    print(f"  R = {R:4.1f}  " + "  ".join(f"{m} {sweep[m][-1]:.5f}" for m in METHODS))
sweep = {m: np.array(v) for m, v in sweep.items()}

# exact R -> infinity limit: two separate H atoms
x_big, N_big = make_grid(max(L, R_SWEEP[-1]/2 + 9), H)
E_H_atom, _ = one_electron(v_ext(1, x_big), H, N_big)


# 1. Energy comparisons
# error of each approximate method against the exact energy, per system
fig, ax = plt.subplots(figsize=(7, 4))
xs = np.arange(len(SOLVERS))
wd = 0.8/len(APPROX)
for i, m in enumerate(APPROX):
    err = [1000*(res[s][m]["E"] - res[s]["exact"]["E"]) for s in SOLVERS]
    bars = ax.bar(xs + (i - (len(APPROX)-1)/2)*wd, err, wd, color=COL[m], label=LAB[m])
    ax.bar_label(bars, fmt="%.1f", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels([TITLE[s] for s in SOLVERS])
ax.set_ylabel("$E_{method} - E_{exact}$ (mHa)")
ax.set_title("Energy error against the exact solution (h = 0.1)")
ax.legend(fontsize=8)
save(fig, "energy_errors.png")

# H2 energy curve and its error
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for m in METHODS:
    axs[0].plot(R_SWEEP, sweep[m], "o-" if m == "exact" else "s--", color=COL[m], label=LAB[m], ms=4)
axs[0].axhline(2*E_H_atom, color="gray", ls=":", label="2 x E(H atom)")
axs[0].set_xlabel("R (bohr)"); axs[0].set_ylabel("total energy at fixed nuclei (Ha)")
axs[0].set_title("H$_2$ energy curve"); axs[0].legend(fontsize=8)
for m in APPROX:
    axs[1].plot(R_SWEEP, sweep[m] - sweep["exact"], "s-", color=COL[m], label=LAB[m], ms=4)
axs[1].axhline(0, color="gray", lw=0.6)
axs[1].set_xlabel("R (bohr)"); axs[1].set_ylabel("$E_{method} - E_{exact}$ (Ha)")
axs[1].set_title("Error grows as the bond stretches (static correlation)"); axs[1].legend(fontsize=8)
save(fig, "H2_energy_curve.png")


# 2. Density comparisons
# top row: n(x) for every method; bottom row: n_method(x) - n_exact(x)
fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for j, name in enumerate(SOLVERS):
    r = res[name]
    for m in METHODS:
        axs[0, j].plot(r[m]["x"], r[m]["n"], color=COL[m], ls="-" if m == "exact" else "--", label=LAB[m])
    for m in APPROX:
        axs[1, j].plot(r[m]["x"], r[m]["n"] - r["exact"]["n"], color=COL[m], label=LAB[m])
    axs[1, j].axhline(0, color="gray", lw=0.6)
    if name == "H2":
        for X in (-R_MAIN/2, R_MAIN/2): # nuclei positions
            axs[0, j].axvline(X, color="gray", lw=0.6, ls=":")
    axs[0, j].set_title(TITLE[name])
    axs[1, j].set_xlabel("x (bohr)"); axs[1, j].set_xlim(-7, 7)
axs[0, 0].set_ylabel("n(x)")
axs[1, 0].set_ylabel("n$_{method}$(x) - n$_{exact}$(x)")
axs[0, 0].legend(fontsize=8, loc="upper right")
save(fig, "densities.png")
