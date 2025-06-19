import numpy as np
import matplotlib.pyplot as plt

logs_folder = "./logs/"
var = input("Enter variable name:")
file = logs_folder + var

data = np.loadtxt(file, delimiter="\t", skiprows=1)
time = data[:, 0]
residuals = data[:, 1:]

plt.plot(time, residuals, "r-o", markersize=2, linewidth=0.5)
plt.xlabel("Time (iteration, RANS)")
plt.ylabel("Residuals")
plt.grid()
plt.yscale("log")
plt.title(f"Residuals for {var}")
plt.savefig(f"./figs/{var}_residuals.png", dpi=300, bbox_inches="tight")
