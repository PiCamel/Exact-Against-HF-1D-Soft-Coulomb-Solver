# Compares the exact and Kohn-Sham (exact exchange = RHF) solvers on H-, He and H2.
# Every number quoted in the report is printed to the terminal by this file; the plots go to results/.
#   1. Atoms and H2 at R = 3.5: energies, correlation energy, energy terms, the two energy routes, HOMO test
#   2. H2 energy curve E(R) and the pair probability
#   3. Grid convergence of He with Richardson extrapolation (plus the one-electron H atom and He+)
#   4. Box-size convergence
#   5. SCF mixing and charge sloshing
#   Plots: energy_errors.png, H2_energy_curve.png, densities.png
# Every method uses the same grid for a given system, so the results can be compared point by point.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg") # write files, no windows
import matplotlib.pyplot as plt

from exact import exact_H_minus, exact_He, exact_H2
from KS import KS_H_minus, KS_He, KS_H2
from common import make_grid, v_ext, one_electron, richardson, convergence_order

METHODS = ["exact", "KS"]
APPROX = METHODS[1:] # the approximate (mean-field) methods

SOLVERS = {"H-": {"exact": exact_H_minus, "KS": KS_H_minus},
           "He": {"exact": exact_He,      "KS": KS_He},
           "H2": {"exact": exact_H2,      "KS": KS_H2}}
Z_ATOM = {"H-": 1, "He": 2} # nuclear charge of the atoms

H = 0.1                 # grid spacing
L = 10.0                # box half-width
R_MAIN = 3.5            # H2 bond length for the single-geometry comparison
R_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
H_GRID = [0.4, 0.2, 0.1, 0.05]                                 # grid study (He)
L_BOX = [6.0, 8.0, 10.0, 12.0, 14.0]                           # box study
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]    # mixing study

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


def table(title, header, rows):
    # prints a plain-text table; every cell is already a string
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    print("\n" + title)
    print("  " + "  ".join(f"{c:>{w}}" for c, w in zip(header, widths)))
    for r in rows:
        print("  " + "  ".join(f"{c:>{w}}" for c, w in zip(r, widths)))


def one_electron_energy(Z, L, h): # one electron around a nucleus of charge Z: the 1D H atom (Z = 1) or He+ (Z = 2)
    x, N = make_grid(L, h)
    eps, _ = one_electron(v_ext(Z, x), h, N)
    return eps


def box_for(R):
    return max(L, R/2 + 9) # the box must grow with R so the density is not squashed by the walls


def pct(delta, ref):
    return 100*abs(delta)/abs(ref)


# =====================================================================
# 1. Atoms and H2 at R = 3.5
# =====================================================================
res = {}
for name, solver in SOLVERS.items():
    extra = {"R": R_MAIN} if name == "H2" else {}
    res[name] = {m: solver[m].run(L=L, h=H, **extra) for m in METHODS}

rows = []
for name in SOLVERS:
    Ex, Eks = res[name]["exact"]["E"], res[name]["KS"]["E"]
    rows.append([name, f"{Ex:.5f}", f"{Eks:.5f}", f"{Eks - Ex:.5f}", f"{pct(Eks - Ex, Ex):.2f}"])
table(f"1a. Total energies (Ha), h = {H}, L = {L}. H2 includes the nuclear repulsion E_nn = w(R).",
      ["system", "E exact", "E KS", "KS - exact (= -E_c)", "delta (%)"], rows)

rows = []
for name in SOLVERS:
    k = res[name]["KS"]
    rows.append([name, f"{k['T_s']:.4f}", f"{k['E_ext']:.4f}", f"{k['E_H']:.4f}", f"{k['E_x']:.4f}",
                 f"{k.get('E_nn', 0.0):.4f}", f"{k['E']:.5f}", f"{k['E_check']:.5f}", f"{abs(k['E'] - k['E_check']):.1e}",
                 str(k["iters"]), f"{k['h']*np.sum(k['n']):.7f}", f"{res[name]['exact']['h']*np.sum(res[name]['exact']['n']):.7f}"])
table("1b. KS energy terms (Ha) and checks",
      ["system", "T_s", "E_ext", "E_H", "E_x", "E_nn", "E (terms)", "E (2eps route)", "|difference|", "iterations",
       "int n (KS)", "int n (exact)"], rows)

E_1e = {name: one_electron_energy(Z, L, H) for name, Z in Z_ATOM.items()} # H atom for H-, He+ for He
rows = []
for name in Z_ATOM:
    I = E_1e[name] - res[name]["exact"]["E"]   # ionisation energy: one-electron system minus two-electron system
    mEps = -res[name]["KS"]["eps"]
    rows.append([name, f"{E_1e[name]:.5f}", f"{I:.4f}", f"{mEps:.4f}", f"{mEps - I:.4f}", f"{pct(mEps - I, I):.2f}"])
table("1c. HOMO test: in exact KS, -eps_HOMO = I",
      ["system", "E(one electron)", "I exact", "-eps KS", "difference", "delta (%)"], rows)

rows = []
for name in SOLVERS:
    r = res[name]
    i0 = len(r["exact"]["x"])//2 # x = 0 is the middle grid point
    rows.append([name, f"{r['exact']['n'].max():.3f}", f"{r['KS']['n'].max():.3f}",
                 f"{r['exact']['n'][i0]:.3f}", f"{r['KS']['n'][i0]:.3f}"])
table("1d. Density peaks", ["system", "max n exact", "max n KS", "n(0) exact", "n(0) KS"], rows)

# =====================================================================
# 2. H2 energy curve and the pair probability
# =====================================================================
sweep = {m: [] for m in METHODS}
E_nn, E_elec = [], [] # nuclear repulsion, and the exact electronic energy without it
for R in R_SWEEP:
    for m in METHODS:
        r = SOLVERS["H2"][m].run(R=R, L=box_for(R), h=H)
        sweep[m].append(r["E"])
        if m == "exact":
            E_nn.append(r["E_nn"]); E_elec.append(r["E_elec"])
sweep = {m: np.array(v) for m, v in sweep.items()}
err = sweep["KS"] - sweep["exact"]
table(f"2a. H2 energy curve (Ha), h = {H}, L = max({L}, R/2 + 9). Total energy = electronic energy + E_nn",
      ["R", "E exact", "E KS", "KS - exact", "delta (%)", "E_nn = w(R)", "exact electronic (no E_nn)"],
      [[f"{R:.1f}", f"{a:.5f}", f"{b:.5f}", f"{b - a:.5f}", f"{pct(b - a, a):.2f}", f"{enn:.5f}", f"{el:.5f}"]
       for R, a, b, enn, el in zip(R_SWEEP, sweep["exact"], sweep["KS"], E_nn, E_elec)])
print(f"  mean absolute error over the {len(R_SWEEP)} distances: {np.mean(np.abs(err)):.4f} Ha")

# exact R -> infinity limit: two separate H atoms
E_H_atom = one_electron_energy(1, box_for(R_SWEEP[-1]), H)
print(f"  2 x E(H atom) (R -> infinity limit): {2*E_H_atom:.5f} Ha")

# probability that both electrons sit on the same side of the molecule (x1 * x2 > 0), H2 at R_MAIN
ex, ks = res["H2"]["exact"], res["H2"]["KS"]
X1, X2 = np.meshgrid(ex["x"], ex["x"], indexing="ij")
same_side = X1*X2 > 0
P_exact = (ex["Psi"]**2 * H*H)[same_side].sum()
P_rhf = (np.outer(ks["phi"]**2, ks["phi"]**2) * H*H)[same_side].sum() # RHF: Psi(x1,x2) = phi(x1) phi(x2)
print(f"\n2b. H2 at R = {R_MAIN}: probability both electrons on the same side: exact {P_exact:.3f}, RHF {P_rhf:.3f}")

# =====================================================================
# 3. Grid convergence (He) and Richardson extrapolation
# =====================================================================
grid = {m: [SOLVERS["He"][m].run(L=L, h=hh)["E"] for hh in H_GRID] for m in METHODS}
rows = []
for i, hh in enumerate(H_GRID):
    ratio = {m: (f"{(grid[m][i-2] - grid[m][i-1])/(grid[m][i-1] - grid[m][i]):.2f}" if i >= 2 else "-") for m in METHODS}
    rows.append([f"{hh}", f"{grid['exact'][i]:.6f}", ratio["exact"], f"{grid['KS'][i]:.6f}", ratio["KS"]])
table(f"3a. Grid convergence of He at L = {L} (ratio of successive differences: 4 means error ~ h^2)",
      ["h", "E exact", "ratio", "E KS", "ratio"], rows)

ext = {} # Richardson extrapolations from (0.2, 0.1) and (0.1, 0.05)
for m in METHODS:
    e = dict(zip(H_GRID, grid[m]))
    ext[m] = {"(0.2,0.1)": richardson(e[0.1], e[0.2]), "(0.1,0.05)": richardson(e[0.05], e[0.1])}
    p = [convergence_order(*grid[m][i:i+3]) for i in range(len(H_GRID) - 2)]
    print(f"  {m:5s}: observed order p = " + ", ".join(f"{v:.2f}" for v in p)
          + f"; Richardson (0.2,0.1) = {ext[m]['(0.2,0.1)']:.6f}, (0.1,0.05) = {ext[m]['(0.1,0.05)']:.6f}"
          + f"; error estimate at h = 0.1: |E(0.1) - E(0.2)|/3 = {abs(e[0.1] - e[0.2])/3*1000:.2f} mHa")

# one-electron systems (H atom, He+) and the extrapolated He reference numbers
one = {Z: [one_electron_energy(Z, L, hh) for hh in (0.1, 0.05)] for Z in (1, 2)}
E_H_ext, E_Hep_ext = richardson(one[1][1], one[1][0]), richardson(one[2][1], one[2][0])
E_ex_ext, E_ks_ext = ext["exact"]["(0.1,0.05)"], ext["KS"]["(0.1,0.05)"]
table("3b. Extrapolated reference values, Richardson from h = 0.1 and 0.05 (Ha)",
      ["quantity", "value"],
      [["one-electron H atom", f"{E_H_ext:.5f}"], ["He+", f"{E_Hep_ext:.5f}"],
       ["He exact", f"{E_ex_ext:.5f}"], ["He KS (= RHF)", f"{E_ks_ext:.5f}"],
       ["E_c = E_exact - E_KS (extrapolated)", f"{E_ex_ext - E_ks_ext:.5f}"],
       ["E_c at h = 0.1", f"{res['He']['exact']['E'] - res['He']['KS']['E']:.5f}"],
       ["I = E(He+) - E(He) (extrapolated)", f"{E_Hep_ext - E_ex_ext:.5f}"]])

# =====================================================================
# 4. Box-size convergence
# =====================================================================
rows = []
for LL in L_BOX:
    he = exact_He.run(L=LL, h=H)["E"]
    hm, hm_ks = exact_H_minus.run(L=LL, h=H)["E"], KS_H_minus.run(L=LL, h=H)
    rows.append([f"{LL}", f"{he:.6f}", f"{hm:.6f}", f"{hm_ks['E']:.6f}", f"{hm_ks['E'] - hm:.5f}",
                 f"{pct(hm_ks['E'] - hm, hm):.2f}", f"{-hm_ks['eps']:.4f}"])
table(f"4. Box-size convergence at h = {H} (Ha)",
      ["L", "He exact", "H- exact", "H- KS", "H- KS - exact", "H- delta (%)", "H- -eps KS"], rows)

# =====================================================================
# 5. SCF mixing and charge sloshing
# =====================================================================
mix = {"He": [KS_He.run(L=L, h=H, alpha=a)["iters"] for a in ALPHAS],
       f"H2 (R={R_MAIN})": [KS_H2.run(R=R_MAIN, L=L, h=H, alpha=a)["iters"] for a in ALPHAS]}
table("5a. SCF iterations to reach ||n_out - n_in|| < 1e-6 against the mixing parameter alpha",
      ["alpha"] + [str(a) for a in ALPHAS], [[k] + [str(v) for v in vals] for k, vals in mix.items()])

slosh = KS_H2.run(R=6.0, L=box_for(6.0), h=H, symmetrize=False, max_iter=300)
fixed = KS_H2.run(R=6.0, L=box_for(6.0), h=H, symmetrize=True)
print(f"\n5b. H2 at R = 6, alpha = 0.4: without symmetrizing: converged = {slosh['converged']} after "
      f"{slosh['iters']} iterations (last error {slosh['errs'][-1]:.2f}); "
      f"with symmetrizing: converged = {fixed['converged']} in {fixed['iters']} iterations\n")

# =====================================================================
# Plots
# =====================================================================
# Energy error of each approximate method against the exact energy, per system
fig, ax = plt.subplots(figsize=(7, 4))
xs = np.arange(len(SOLVERS))
wd = 0.8/len(APPROX)
for i, m in enumerate(APPROX):
    e_m = [1000*(res[s][m]["E"] - res[s]["exact"]["E"]) for s in SOLVERS]
    bars = ax.bar(xs + (i - (len(APPROX)-1)/2)*wd, e_m, wd, color=COL[m], label=LAB[m])
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

# Densities. Top row: n(x) for every method; bottom row: n_method(x) - n_exact(x)
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
