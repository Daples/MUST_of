import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {"text.usetex": True, "font.family": "Computer Modern Roman", "font.size": 22}
)

coarse_case = "coarse"
med_case = "med"
fine_case = "fine"

cases = [coarse_case, med_case, fine_case]

colors = ["dodgerblue", "darkorange", "red"]
labels = ["Coarse", "Medium", "Fine"]

# for color, case in zip(colors, cases):
#     print(case)

#     # Load data
#     mesh = pv.read(f"./{case}_MUST/VTK/{case}_MUST_4000/internal.vtu")

#     # Plotting histograms
#     _, ax = plt.subplots(1, 1, figsize=(8, 6))
#     ax.hist(np.log10(mesh["div(U)"] + 1e-15), bins=100, ec="white", fc=color, density=True)
#     ax.axvline(x=0, color="k", alpha=0.8)

#     ax.set_xlabel("$\\mathrm{log}_{10}(\\partial_{i}\\overline{u}_i + 10^{-15})$")
#     ax.set_ylabel("Density")
#     ax.set_yticklabels([])

#     plt.savefig(f"figs/{case}_div_hist.pdf", bbox_inches="tight")
#     plt.clf()

_, ax = plt.subplots(1, 1, figsize=(10, 6))
xs = []
for color, case, case_name in zip(colors, cases, labels):
    print(case)

    # Load data
    mesh = pv.read(f"./{case}_MUST/VTK/{case}_MUST_4000/internal.vtu")

    xs.append(np.log10(mesh["div(U)"] + 1e-15))

# Plotting histograms
ax.hist(xs, bins=100, stacked=True, fc=colors, density=True, label=labels, alpha=1)
ax.axvline(x=0, color="k", alpha=0.8)

ax.set_xlabel("$\\mathrm{log}_{10}(\\partial_{i}\\overline{u}_i + 10^{-15})$")
ax.set_ylabel("Density")
ax.set_yticklabels([])
ax.legend()
ax.grid(alpha=0.4)

plt.savefig(f"figs/div_hist.pdf", bbox_inches="tight")