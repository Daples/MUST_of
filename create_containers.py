from utils.rotate_mesh import rotate_geometry
from stl import mesh

import numpy as np
import string

# Constants
CONTAINER_HEIGHT = 2.54
VIP_CAR_HEIGHT = 3.51
LETTERS = string.ascii_uppercase[:12][::-1]
NUMBERS = string.digits[:10][::-1]
STL_FOLDER = "constant/triSurface/"
H_max = 12.2

# Create a matrix of letters and numbers
matrix = np.array([l + n for n in NUMBERS for l in LETTERS]).reshape(
    (len(NUMBERS), len(LETTERS))
)

# Create a mask to select specific containers
mask = np.loadtxt("sources/selection.txt", delimiter=" ", dtype=str)
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
xmin = np.min(bbox_vertices[:, 0]) - l
xmax = np.max(bbox_vertices[:, 0]) + L
ymin = np.min(bbox_vertices[:, 1]) - l
ymax = np.max(bbox_vertices[:, 1]) + l
zmin = np.min(bbox_vertices[:, 2])
zmax = np.max(bbox_vertices[:, 2]) + lz

print(
    f"Bounding box (guidelines):\n x: [{xmin}, {xmax}],\n y: [{ymin}, {ymax}],\n z: [{zmin}, {zmax}]"
)
