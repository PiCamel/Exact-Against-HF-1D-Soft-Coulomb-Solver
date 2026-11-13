# Runs every analysis file and compares them.
# Outputs: figures/*.png and results/results.md (all tables), plus a printed summary.
#
#   exact_H_minus.py  exact two-electron solve, one nucleus (H- for Z = 1, He for Z = 2)
#   exact_H2.py       exact two-electron solve, two nuclei (H2)
#   KS_H_minus.py     Kohn-Sham with exact exchange (= restricted Hartree-Fock), one nucleus
#   KS_H2.py          Kohn-Sham with exact exchange (= restricted Hartree-Fock), H2
#   LDA_H_minus.py    Kohn-Sham with LDA exchange, one nucleus      (used only once it defines run())
#   LDA_H2.py         Kohn-Sham with LDA exchange, H2               (used only once it defines run())
#
# Every method uses the same grid for a given system, so energies can be compared directly
# and the variational check E_exact <= E_KS is meaningful.
# Runtime: about a minute (the exact solves on the larger grids dominate).

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg") # write files, no windows
import matplotlib.pyplot as plt

import exact_H_minus, exact_H2, KS_H_minus, KS_H2, LDA_H_minus, LDA_H2
from common import make_grid, v_ext, one_electron

# The LDA files are switched on automatically as soon as both define run().
HAS_LDA = hasattr(LDA_H_minus, "run") and hasattr(LDA_H2, "run")
METHODS = ["exact", "KS"] + (["LDA"] if HAS_LDA else [])
MF = [m for m in METHODS if m != "exact"] # mean-field (Kohn-Sham) methods

H = 0.1          # grid spacing used everywhere unless a study varies it
L_ATOM = 10.0    # box half-width for H- and He
R_MAIN = 3.5     # H2 geometry used for the density plots
L_H2 = 10.0      # box for H2 at R_MAIN
R_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]

FIG = "figures"
OUT = "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

COL = {"exact": "#1f1f1f", "KS": "#534AB7", "LDA": "#D85A30"}
LAB = {"exact": "exact", "KS": "KS exact exchange (= RHF)", "LDA": "KS LDA exchange"}
ATOM_SOLVER = {"exact": exact_H_minus, "KS": KS_H_minus, "LDA": LDA_H_minus}
H2_SOLVER = {"exact": exact_H2, "KS": KS_H2, "LDA": LDA_H2}
report = [] # lines of results.md


def box_for(R):
    return max(L_H2, R/2 + 9) # the box must grow with R so the density is not squashed by the walls


def one_electron_energy(Z, L, h=H):
    # energy of a single electron around a nucleus of charge Z: the 1D H atom (Z = 1) or He+ (Z = 2)
    x, N = make_grid(L, h)
    eps, _ = one_electron(v_ext(Z, x), h, N)
    return eps


def save(fig, name, legend_below=None):
    if legend_below is not None: # shared legend under a row of panels
        handles, labels = legend_below.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)
    print("  saved", os.path.join(FIG, name))


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|"*len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def fmt(v, d=5):
    return f"{v:.{d}f}"


t0 = time.time()
print("LDA files", "found: including LDA" if HAS_LDA else "not written yet: skipping LDA")

# =====================================================================
# 1. Atoms: H- (Z = 1) and He (Z = 2), every method on the same grid
# =====================================================================
print("1. atoms")
atoms = {}
for name, Z in (("H-", 1), ("He", 2)):
    res = {m: ATOM_SOLVER[m].run(Z=Z, L=L_ATOM, h=H) for m in METHODS}
    E1 = one_electron_energy(Z, L_ATOM) # H atom or He+
    res.update({"Z": Z, "E_1e": E1, "I": E1 - res["exact"]["E"]})
    atoms[name] = res
    print(f"  {name}: " + "  ".join(f"{m} {res[m]['E']:.5f}" for m in METHODS))

# =====================================================================
# 2. H2 at R = 3.5
# =====================================================================
print("2. H2 at R =", R_MAIN)
h2 = {m: H2_SOLVER[m].run(R=R_MAIN, L=L_H2, h=H) for m in METHODS}

# probability that both electrons are on the same side of the molecule (x1 * x2 > 0)
x = h2["exact"]["x"]
X1, X2 = np.meshgrid(x, x, indexing="ij")
same_side = X1*X2 > 0
P_ex_map = h2["exact"]["Psi"]**2 * H*H
P_exact = P_ex_map[same_side].sum()
phi = h2["KS"]["phi"]
P_rhf_map = np.outer(phi**2, phi**2) * H*H # RHF: Psi(x1,x2) = phi(x1) phi(x2)
P_rhf = P_rhf_map[same_side].sum()

# =====================================================================
# 3. H2 dissociation sweep
# =====================================================================
print("3. H2 dissociation sweep")
sweep = {m: [] for m in METHODS}
for R in R_SWEEP:
    L = box_for(R)
    for m in METHODS:
        sweep[m].append(H2_SOLVER[m].run(R=R, L=L, h=H)["E"])
    print(f"  R = {R:4.1f}  " + "  ".join(f"{m} {sweep[m][-1]:.5f}" for m in METHODS))
sweep = {m: np.array(v) for m, v in sweep.items()}
R_arr = np.array(R_SWEEP)
E_dissoc = 2*one_electron_energy(1, box_for(max(R_SWEEP))) # exact R -> infinity limit: two separate H atoms

# =====================================================================
# 4. Grid convergence (He): energy error goes as h^2, Richardson extrapolation
# =====================================================================
print("4. grid convergence (He)")
hs = [0.2, 0.1, 0.05]
conv = {m: [ATOM_SOLVER[m].run(Z=2, L=L_ATOM, h=hh)["E"] for hh in hs] for m in METHODS}
extrap = {m: conv[m][2] + (conv[m][2] - conv[m][1])/3 for m in METHODS} # E ~ E_h + (E_h - E_2h)/3

# =====================================================================
# 5. SCF mixing study: iterations to converge vs alpha
# =====================================================================
print("5. mixing study")
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mix_runs = {"KS He": lambda a: KS_H_minus.run(Z=2, L=L_ATOM, h=H, alpha=a),
            "KS H2": lambda a: KS_H2.run(R=R_MAIN, L=L_H2, h=H, alpha=a)}
if HAS_LDA:
    mix_runs["LDA He"] = lambda a: LDA_H_minus.run(Z=2, L=L_ATOM, h=H, alpha=a)
mix = {}
for key, f in mix_runs.items():
    mix[key] = []
    for a in alphas:
        r = f(a)
        mix[key].append(r["iters"] if r["converged"] else np.nan)

# charge sloshing: stretched H2 without forcing the density to be symmetric
slosh = KS_H2.run(R=6.0, L=box_for(6.0), h=H, symmetrize=False, max_iter=300)
nosl = KS_H2.run(R=6.0, L=box_for(6.0), h=H, symmetrize=True)

# =====================================================================
# 6. H-: is the second electron bound? (box dependence of the HOMO)
# =====================================================================
print("6. H- box dependence")
boxes = [8.0, 10.0, 12.0, 15.0, 20.0]
box = {"KS": [KS_H_minus.run(Z=1, L=LL, h=H) for LL in boxes]}
if HAS_LDA:
    box["LDA"] = [LDA_H_minus.run(Z=1, L=LL, h=H, alpha=0.1, max_iter=3000) for LL in boxes]

print(f"all calculations done in {time.time()-t0:.0f} s, making figures")

# =====================================================================
# Figures
# =====================================================================
SYSTEMS = (("H$^-$", atoms["H-"]), ("He", atoms["He"]), (f"H$_2$, R = {R_MAIN}", h2))

# Fig 1: densities
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
for ax, (title, res) in zip(axs, SYSTEMS):
    for m in METHODS:
        ax.plot(res[m]["x"], res[m]["n"], color=COL[m], ls="-" if m == "exact" else "--", label=LAB[m])
    ax.set_title(title); ax.set_xlabel("x (bohr)"); ax.set_xlim(-7, 7)
axs[0].set_ylabel("n(x)")
for X in (-R_MAIN/2, R_MAIN/2):
    axs[2].axvline(X, color="gray", lw=0.6, ls=":")
save(fig, "fig1_densities.png", legend_below=axs[0])

# Fig 2: density errors relative to exact
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
for ax, (title, res) in zip(axs, SYSTEMS):
    for m in MF:
        ax.plot(res[m]["x"], res[m]["n"] - res["exact"]["n"], color=COL[m], label=LAB[m])
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(title); ax.set_xlabel("x (bohr)"); ax.set_xlim(-7, 7)
axs[0].set_ylabel("n(x) - n$_{exact}$(x)")
save(fig, "fig2_density_errors.png", legend_below=axs[0])

# Fig 3: energy terms for He
fig, ax = plt.subplots(figsize=(7, 4))
terms = ["T_s", "E_ext", "E_H", "E_x"]
xs = np.arange(len(terms))
wd = 0.8/len(MF)
for i, m in enumerate(MF):
    ax.bar(xs + (i - (len(MF)-1)/2)*wd, [atoms["He"][m][t] for t in terms], wd, color=COL[m], label=LAB[m])
ax.axhline(0, color="gray", lw=0.6)
ax.set_xticks(xs); ax.set_xticklabels(["$T_s$", "$E_{ext}$", "$E_H$", "$E_x$"]); ax.set_ylabel("energy (Ha)")
ax.set_title("He: Kohn-Sham energy terms"); ax.legend(fontsize=8)
save(fig, "fig3_He_energy_breakdown.png")

# Fig 4: H2 dissociation curves
fig, ax = plt.subplots(figsize=(7, 4.5))
for m in METHODS:
    ax.plot(R_arr, sweep[m], "o-" if m == "exact" else "s--", color=COL[m], label=LAB[m], ms=4)
ax.axhline(E_dissoc, color="gray", ls=":", label="2 x E(H atom)")
ax.set_xlabel("R (bohr)"); ax.set_ylabel("total energy at fixed nuclei (Ha)")
ax.set_title("H$_2$ dissociation: mean field fails at large R"); ax.legend(fontsize=8)
save(fig, "fig4_H2_dissociation.png")

# Fig 5: errors along the dissociation curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(R_arr, sweep["exact"] - sweep["KS"], "s-", color=COL["KS"], label="$E_c = E_{exact} - E_{KS}$ (exact correlation)")
if HAS_LDA:
    ax.plot(R_arr, sweep["exact"] - sweep["LDA"], "s-", color=COL["LDA"], label="$E_{exact} - E_{LDA}$")
ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel("R (bohr)"); ax.set_ylabel("energy difference (Ha)")
ax.set_title("Static correlation grows as the bond stretches"); ax.legend(fontsize=8)
save(fig, "fig5_H2_errors_vs_R.png")

# Fig 6: pair probability, exact vs RHF product (H2 at R_MAIN)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
vmax = max(P_ex_map.max(), P_rhf_map.max())
for ax, mp, title in ((axs[0], P_ex_map, f"exact $|\\Psi(x_1,x_2)|^2$\nP(same side) = {P_exact:.2f}"),
                      (axs[1], P_rhf_map, f"RHF $\\varphi(x_1)^2\\varphi(x_2)^2$\nP(same side) = {P_rhf:.2f}")):
    im = ax.imshow(mp.T, origin="lower", extent=[x[0], x[-1], x[0], x[-1]], vmin=0, vmax=vmax, cmap="magma")
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_xlabel("$x_1$ (bohr)"); ax.set_ylabel("$x_2$ (bohr)"); ax.set_title(title, fontsize=10)
fig.colorbar(im, ax=axs, shrink=0.85)
fig.suptitle(f"H$_2$ at R = {R_MAIN}: where are the two electrons together?", fontsize=11)
fig.savefig(os.path.join(FIG, "fig6_H2_pair_probability.png"), dpi=150); plt.close(fig)
print("  saved", os.path.join(FIG, "fig6_H2_pair_probability.png"))

# Fig 7: exchange potentials (He)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(atoms["He"]["KS"]["x"], atoms["He"]["KS"]["v_XC"], color=COL["KS"], label="exact exchange $-v_H/2$")
if HAS_LDA:
    ax.plot(atoms["He"]["LDA"]["x"], atoms["He"]["LDA"]["v_XC"], color=COL["LDA"], label="LDA $v_x(n(x))$")
ax.set_xlabel("x (bohr)"); ax.set_ylabel("$v_x(x)$ (Ha)"); ax.set_xlim(-10, 10)
ax.set_title("He: exchange potential"); ax.legend(fontsize=8)
save(fig, "fig7_He_exchange_potentials.png")

# Fig 8: SCF convergence histories and charge sloshing
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for m in MF:
    axs[0].semilogy(atoms["He"][m]["errs"], color=COL[m], label=f"He, {LAB[m]}")
axs[0].semilogy(h2["KS"]["errs"], color=COL["KS"], ls="--", label=f"H$_2$ R = {R_MAIN}, {LAB['KS']}")
axs[0].set_xlabel("iteration"); axs[0].set_ylabel("|n$_{new}$ - n|"); axs[0].set_title("SCF convergence ($\\alpha$ = 0.4)"); axs[0].legend(fontsize=8)
axs[1].semilogy(slosh["errs"], color="#993C1D", label="no symmetry imposed (sloshes)")
axs[1].semilogy(nosl["errs"], color=COL["KS"], label="density symmetrized")
axs[1].set_xlabel("iteration"); axs[1].set_title("Stretched H$_2$ (R = 6): charge sloshing"); axs[1].legend(fontsize=8)
save(fig, "fig8_scf_convergence.png")

# Fig 9: mixing study
fig, ax = plt.subplots(figsize=(7, 4))
style = {"KS He": (COL["KS"], "-"), "LDA He": (COL["LDA"], "-"), "KS H2": (COL["KS"], "--")}
for key in mix:
    c, ls = style[key]
    ax.plot(alphas, mix[key], "o" + ls, color=c, label=key.replace("H2", f"H$_2$ (R = {R_MAIN})"), ms=4)
ax.set_xlabel("mixing parameter $\\alpha$"); ax.set_ylabel("iterations to converge")
ax.set_title("SCF mixing study"); ax.legend(fontsize=8)
save(fig, "fig9_mixing_study.png")

# Fig 10: grid convergence
fig, ax = plt.subplots(figsize=(7, 4))
for m in METHODS:
    ax.plot(np.array(hs)**2, (np.array(conv[m]) - extrap[m])*1000, "o-", color=COL[m], label=LAB[m])
ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel("$h^2$ (bohr$^2$)"); ax.set_ylabel("$E_h - E_{extrapolated}$ (mHa)")
ax.set_title("He grid convergence: error $\\propto h^2$"); ax.legend(fontsize=8)
save(fig, "fig10_grid_convergence.png")

# Fig 11: H- HOMO eigenvalue vs box size
fig, ax = plt.subplots(figsize=(7, 4))
for m in box:
    ax.plot(boxes, [r["eps"] for r in box[m]], "s-", color=COL[m], label=f"{LAB[m]}: HOMO $\\varepsilon$")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("box half-width L (bohr)"); ax.set_ylabel("HOMO eigenvalue (Ha)")
ax.set_title("H$^-$: is the second electron bound? ($\\varepsilon < 0$ means bound)"); ax.legend(fontsize=8)
save(fig, "fig11_Hminus_box.png")

# =====================================================================
# Tables -> results/results.md
# =====================================================================
report += ["# Results", "",
           f"All energies in Hartree. Grid spacing h = {H} unless stated. Box half-width L = {L_ATOM} for atoms.",
           "LDA results " + ("included." if HAS_LDA else "not included yet (LDA_H_minus.py / LDA_H2.py have no run() yet)."), ""]

report += [f"## Atoms and H2 at R = {R_MAIN}", ""]
head = ["system"] + [f"E {m}" for m in METHODS] + [f"exact - {m}" for m in MF] + ["I exact"] + [f"-eps {m}" for m in MF]
rows = []
for name, res in (("H-", atoms["H-"]), ("He", atoms["He"]), (f"H2 (R = {R_MAIN})", h2)):
    I = fmt(res["I"], 4) if "I" in res else "-"
    rows.append([name] + [fmt(res[m]["E"]) for m in METHODS] + [fmt(res["exact"]["E"] - res[m]["E"]) for m in MF]
                + [I] + [fmt(-res[m]["eps"], 4) for m in MF])
report += md_table(head, rows)
report += ["", "exact - KS is the exact correlation energy E_c. I exact = E(one-electron system) - E(two-electron system) "
           "is the true ionization energy; in exact Kohn-Sham, -eps of the highest orbital equals I.", ""]

report += ["## Energy terms", ""]
rows = []
for name, res in (("H-", atoms["H-"]), ("He", atoms["He"]), (f"H2 R={R_MAIN}", h2)):
    for m in MF:
        r = res[m]
        rows.append([name, m, fmt(r["T_s"], 4), fmt(r["E_ext"], 4), fmt(r["E_H"], 4), fmt(r["E_x"], 4),
                     fmt(-0.5*r["E_H"], 4), fmt(r.get("E_nn", 0.0), 4), fmt(r["E"]), fmt(r["E_check"]), str(r["iters"])])
report += md_table(["system", "method", "T_s", "E_ext", "E_H", "E_x", "-E_H/2", "E_nn", "E (terms)", "E (eigenvalue route)", "iterations"], rows)
report += ["", "E (terms) and E (eigenvalue route) are two independent ways of computing the same energy; they must agree.", ""]

report += [f"## H2 pair probability at R = {R_MAIN}", "",
           f"Probability that both electrons are on the same side of the molecule: exact {P_exact:.3f}, RHF {P_rhf:.3f}.", ""]

report += ["## H2 dissociation curve", "", f"Exact limit R -> infinity: 2 x E(H atom) = {E_dissoc:.5f}", ""]
rows = [[f"{R:.1f}"] + [fmt(sweep[m][i]) for m in METHODS] + [fmt(sweep["exact"][i] - sweep[m][i]) for m in MF]
        for i, R in enumerate(R_SWEEP)]
report += md_table(["R"] + METHODS + [f"exact - {m}" for m in MF], rows)
report += [""]

report += ["## Grid convergence (He)", ""]
rows = [[f"{hh}"] + [fmt(conv[m][i], 6) for m in METHODS] for i, hh in enumerate(hs)]
rows.append(["extrapolated"] + [fmt(extrap[m], 6) for m in METHODS])
report += md_table(["h"] + METHODS, rows)
report += ["", "Extrapolated with E = E_h + (E_h - E_2h)/3 from h = 0.1 and 0.05.", ""]

report += ["## SCF mixing study (iterations to converge)", ""]
rows = [[f"{a}"] + [("did not converge" if np.isnan(mix[k][i]) else str(int(mix[k][i]))) for k in mix] for i, a in enumerate(alphas)]
report += md_table(["alpha"] + list(mix), rows)
report += ["", f"Charge sloshing, H2 at R = 6 without symmetrizing the density: converged = {slosh['converged']} "
           f"after {slosh['iters']} iterations; with symmetrizing: converged in {nosl['iters']} iterations.", ""]

report += ["## H-: box dependence of the HOMO eigenvalue", ""]
rows = [[f"{LL}"] + sum([[fmt(box[m][i]["E"]), fmt(box[m][i]["eps"], 4)] for m in box], []) for i, LL in enumerate(boxes)]
report += md_table(["L"] + sum([[f"E {m}", f"eps {m}"] for m in box], []), rows)
report += ["", "A negative HOMO eigenvalue means the electron is bound. A positive one means it is held only by the box "
           "walls, so its energy depends on the box size.", ""]

with open(os.path.join(OUT, "results.md"), "w") as f:
    f.write("\n".join(report) + "\n")
print("  saved", os.path.join(OUT, "results.md"))
print("\n".join(report))
print(f"total time {time.time()-t0:.0f} s")
