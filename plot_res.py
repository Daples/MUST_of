# Inspired on code by Clara Garcia Sanchez - 03/02/2022
# --------------------------------------------------------------------------------------------------------#
# Libraries
# --------------------------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------------#
# CFD results
# --------------------------------------------------------------------------------------------------------#
iteration = []
p = []
Ux = []
Uy = []
Uz = []
k = []
epsilon = []
nfolder = "./postProcessing/solverInfo/2"
nfile = "solverInfo.dat"
with open(nfolder + "/" + nfile, encoding="utf8", errors="ignore") as f:
    lines = f.readlines()
    for line in lines[2:]:
        fields = np.array(line.split())
        iteration.append(float(fields[0]))
        p.append(float(fields[24]))  # p_initial
        Ux.append(float(fields[3]))  # Ux_final
        Uy.append(float(fields[6]))  # Uy_final
        Uz.append(float(fields[9]))  # Uz_final
        k.append(float(fields[19]))  # k_final
        epsilon.append(float(fields[14]))  # epsilon_final
    f.close()

# --------------------------------------------------------------------------------------------------------#
# Plotting
# --------------------------------------------------------------------------------------------------------#
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111)

s = slice(None, 1000)
ax1.plot(iteration[s], p[s], "k-")
ax1.plot(iteration[s], Ux[s], "b-")
ax1.plot(iteration[s], Uy[s], "c-")
ax1.plot(iteration[s], Uz[s], "-", color="orange")
ax1.plot(iteration[s], k[s], "r-")
ax1.plot(iteration[s], epsilon[s], "g-")

ax1.set_xlabel("Time (iteration, RANS)", fontsize=14)
ax1.set_ylabel("Residuals")
ax1.legend(["p", "Ux", "Uy", "Uz", "k", "epsilon"], loc="upper right")
ax1.set_yscale("log")
ax1.grid()

plt.savefig(f"./figs/residuals.pdf", bbox_inches="tight")
plt.show()
