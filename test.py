import pyvista as pv
import numpy as np

# Load data
mesh = pv.read("VTK/MUST_890/internal.vtu")

# Check available scalar fields
print(mesh.array_names)

# Compute norm of U
velocity = mesh["U"]
U_mag = np.linalg.norm(velocity, axis=1)
mesh["U_mag"] = U_mag  # Add to mesh
scalar_to_plot = "U_mag"

# Create slice
z_value = 1.75  # Change
slice_plane = mesh.slice(normal="z", origin=(0, 0, z_value))

# Plot
plotter = pv.Plotter()
plotter.add_mesh(slice_plane, scalars=scalar_to_plot, cmap="coolwarm")
plotter.add_axes()
plotter.show_bounds(grid="front", location="outer")
plotter.show(title=f"Slice plot of {scalar_to_plot}")
