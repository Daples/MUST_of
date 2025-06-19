import trimesh
import numpy as np
import argparse


def rotate_geometry(input_file, output_file, angle_deg, axis="z", center=None):
    # Load the mesh
    mesh = trimesh.load(input_file)

    # Determine rotation center
    if center is None:
        center = mesh.centroid  # type: ignore
    center = np.array(center)

    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)

    # Define rotation axis
    axis_dict = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}

    if axis not in axis_dict:
        raise ValueError(f"Axis must be 'x', 'y', or 'z'. Got '{axis}'.")

    rotation_axis = axis_dict[axis]

    # Create the rotation matrix
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle_rad, rotation_axis, point=center
    )

    # Apply the transformation
    mesh.apply_transform(rotation_matrix)

    # Export the rotated mesh
    mesh.export(output_file)
    print(f"Rotated mesh saved to {output_file}")

    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotate a geometry using trimesh.")
    parser.add_argument("input", help="Input mesh file (e.g., .stl, .obj, .ply)")
    parser.add_argument("output", help="Output mesh file")
    parser.add_argument("angle", type=float, help="Rotation angle in degrees")
    parser.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="z",
        help="Rotation axis (default: z)",
    )
    parser.add_argument(
        "--center", nargs=3, type=float, help="Optional center of rotation (x y z)"
    )

    args = parser.parse_args()

    rotate_geometry(
        args.input, args.output, args.angle, axis=args.axis, center=args.center
    )
