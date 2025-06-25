from utils.rotate_mesh import rotate_geometry
from stl import mesh
from numpy.dtypes import StringDType

import numpy as np
import string
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Constants
CONTAINER_HEIGHT = 2.54
VIP_CAR_HEIGHT = 3.51
LETTERS = string.ascii_uppercase[:12][::-1]
NUMBERS = string.digits[:10][::-1]
STL_FOLDER = "constant/triSurface/"
H_max = 12.2

# Create a matrix of letters and numbers
matrix = (
    np.array([l + n for n in NUMBERS for l in LETTERS])
    .reshape((len(NUMBERS), len(LETTERS)))
    .astype(StringDType())
)
matrix[np.where(matrix == "H5")] = "VIP car H5"

# Create a mask to select specific containers
mask = np.loadtxt("sources/selection_all.txt", delimiter=" ", dtype=str)
mask = mask[1:, :-1].astype(int)
masked_tags = matrix[np.where(mask == 1)].flatten().tolist()

# The containers are not rotated
all_data = np.loadtxt("sources/containers.txt", delimiter="\t", dtype=str)
all_tags = all_data[:, 0].astype(str)
indices = np.where(np.isin(all_tags, masked_tags))[0]

# Select containers to use
coords = all_data[indices, 1:].astype(float)
tags = all_tags[indices].tolist()
n_objects = coords.shape[0]

# Define the 12 triangles composing a cube
faces = np.array(
    [
        [0, 3, 1],
        [1, 3, 2],
        [0, 4, 7],
        [0, 7, 3],
        [4, 5, 6],
        [4, 6, 7],
        [5, 1, 2],
        [5, 2, 6],
        [2, 3, 6],
        [3, 7, 6],
        [0, 1, 5],
        [0, 5, 4],
    ]
)

# Create the mesh
count_obj = 0
cube = mesh.Mesh(np.zeros(faces.shape[0] * n_objects, dtype=mesh.Mesh.dtype))
for tag, xys in zip(tags, coords):
    xs = xys[0::2]
    ys = xys[1::2]

    # Reorder coordinates | Specific for how this data is structured
    reordering = [[2, 0, 1, 3]]
    xs = xs[reordering]
    ys = ys[reordering]

    # Repeat for height
    xs = np.tile(xs, 2)
    ys = np.tile(ys, 2)

    # Add height to z-coordinates
    height = CONTAINER_HEIGHT if "VIP" not in tag else VIP_CAR_HEIGHT
    zs = np.zeros_like(xs) + np.repeat((0, 1), 4) * height
    vertices = np.zeros((8, 3))

    vertices[:, 0] = xs
    vertices[:, 1] = ys
    vertices[:, 2] = zs

    # Add to the mesh
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i + count_obj][j] = vertices[f[j], :]  # type: ignore

    # Update the count of cubes for the next container
    count_obj += faces.shape[0]

# Write the mesh to .stl file
cube.save(STL_FOLDER + "buildings_og.stl")  # type: ignore

# Rotate the mesh
deg = 45
mesh = rotate_geometry(
    STL_FOLDER + "buildings_og.stl",
    STL_FOLDER + "buildings.stl",
    deg,
    axis="z",
)

bbox_vertices = mesh.bounding_box.vertices  # type: ignore

# Calculate the bounding box dimensions according to the guidelines
l = 5 * H_max
L = 15 * H_max
lz = 5 * CONTAINER_HEIGHT
xmin = np.min(bbox_vertices[:, 0])
xmax = np.max(bbox_vertices[:, 0])
ymin = np.min(bbox_vertices[:, 1])
ymax = np.max(bbox_vertices[:, 1])
zmin = np.min(bbox_vertices[:, 2])
zmax = np.max(bbox_vertices[:, 2])

xmin_guidelines = xmin - l
xmax_guidelines = xmax + L
ymin_guidelines = ymin - l
ymax_guidelines = ymax + l
zmin_guidelines = zmin
zmax_guidelines = zmax + l

# Calculate the blockage ratio
area_blockage = (ymax - ymin) * CONTAINER_HEIGHT
area_domain = (ymax_guidelines - ymin_guidelines) * (zmax_guidelines - zmin_guidelines)
blockage_ratio = area_blockage / area_domain

# Calculate directional blockage ratio
l_building = ymax - ymin
l_domain = ymax_guidelines - ymin_guidelines
blockage_ratio_l = l_building / l_domain

h_building = zmax - zmin
h_domain = zmax_guidelines - zmin_guidelines
blockage_ratio_h = h_building / h_domain

print(
    "Bounding box (guidelines):\n"
    + f"x: [{xmin_guidelines}, {xmax_guidelines}],\n"
    + f"y: [{ymin_guidelines}, {ymax_guidelines}],\n"
    + f"z: [{zmin_guidelines}, {zmax_guidelines}],\n"
    + f"Blockage ratio: {blockage_ratio:.2%}\n"
    + f"Blockage ratio (Ly): {blockage_ratio_l:.2%}\n"
    + f"Blockage ratio (Hz): {blockage_ratio_h:.2%}"
)

# Plot the bounding boxes
fig = plt.figure(figsize=(15, 10))
ax = fig.subplots(1, 1)

ax.plot(
    mesh.vertices[:, 0],  # type: ignore
    mesh.vertices[:, 1],  # type: ignore
    "o",
    markersize=1,
    color="black",
    label="Vertices",
)
bbox = patches.Rectangle(
    (xmin, ymin),
    xmax - xmin,
    ymax - ymin,
    linewidth=1,
    edgecolor="red",
    facecolor="none",
    label="Bounding box",
)
ax.add_patch(bbox)

guidelines = patches.Rectangle(
    (xmin_guidelines, ymin_guidelines),
    xmax_guidelines - xmin_guidelines,
    ymax_guidelines - ymin_guidelines,
    linewidth=1,
    edgecolor="blue",
    facecolor="none",
    label="Guidelines",
)
ax.add_patch(guidelines)

plt.grid()
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()

plt.show()
