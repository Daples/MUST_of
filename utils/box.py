from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Any
from matplotlib.axes import Axes

import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class Box:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    dims: ClassVar[str] = "xyz"

    def compute_area(self, axis: int) -> float:
        """Compute the area of the box.

        Parameters
        ----------
        dim : int
            The dimension for which to compute the area.

        Returns
        -------
        float
            The area of the box in the specified dimension.
        """

        new_dims = self.dims.replace(self.dims[axis], "")
        area = 1
        for dim in new_dims:
            area *= self.get_length(dim)
        return area

    def get_length(self, dim: str) -> float:
        """Get the length of the box in a specific dimension.

        Parameters
        ----------
        dim : str
            The dimension for which to get the length.

        Returns
        -------
        float
            The length of the box in the specified dimension.
        """

        return self.__getattribute__(f"{dim}max") - self.__getattribute__(f"{dim}min")

    def expand(
        self, new_name: str, l_xm: float, l_xp: float, l_y: float, l_z: float
    ) -> Box:
        """Create a guidelines box based on the current box."""

        return Box(
            name=new_name,
            xmin=self.xmin - l_xm,
            xmax=self.xmax + l_xp,
            ymin=self.ymin - l_y,
            ymax=self.ymax + l_y,
            zmin=self.zmin,
            zmax=self.zmax + l_z,
        )

    def plot(self, ax: Axes, color: str, **kwargs: Any) -> Axes:
        """"""

        bbox = patches.Rectangle(
            (self.xmin, self.ymin),
            self.get_length("x"),
            self.get_length("y"),
            linewidth=1,
            edgecolor=color,
            facecolor="none",
            label=self.name,
            **kwargs,
        )
        ax.add_patch(bbox)
        return ax

    def evaluate(self, bbox_buildings: Box, L_max: float, flow_dim: int) -> None:
        """"""

        flow = self.dims[flow_dim]
        z = self.dims[2]

        # Calculate the blockage ratio
        area_blockage = bbox_buildings.compute_area(axis=flow_dim)
        area_domain = self.compute_area(axis=flow_dim)
        blockage_ratio = area_blockage / area_domain

        # Calculate directional blockage ratio
        l_buildings = bbox_buildings.get_length(flow)
        l_domain = self.get_length(flow)
        blockage_ratio_l = l_buildings / l_domain

        h_buildings = bbox_buildings.get_length(z)
        h_domain = self.get_length(z)
        blockage_ratio_h = h_buildings / h_domain

        # Directional blockage ratio (single container at largest blockage, 45°)
        blockage_ratio_container = L_max / l_domain

        print(
            f"\nBounding box ({self.name}):\n"
            + f"x: [{self.xmin:.2f}, {self.xmax:.2f}],\n"
            + f"y: [{self.ymin:.2f}, {self.ymax:.2f}],\n"
            + f"z: [{self.zmin:.2f}, {self.zmax:.2f}],\n"
            + f"Blockage ratio: {blockage_ratio:.2%}\n"
            + f"Blockage ratio (L{flow}, all buildings): {blockage_ratio_l:.2%}\n"
            + f"Blockage ratio (H{z}, all buildings): {blockage_ratio_h:.2%}\n"
            + f"Blockage ratio (Single container): {blockage_ratio_container:.2%}"
        )

    def get_expansion(
        self, box_buildings: Box, H_max: float | None = None
    ) -> tuple[float, float, float, float]:
        """"""

        l_xm = abs(self.xmin - box_buildings.xmin)
        l_xp = abs(self.xmax - box_buildings.xmax)
        l_y = abs(self.ymax - box_buildings.ymax)
        l_z = abs(self.zmax - box_buildings.zmax)

        if H_max is not None:
            print(f"l_xm: {l_xm / H_max:.2f}")
            print(f"l_xp: {l_xp / H_max:.2f}")
            print(f"l_y: {l_y / H_max:.2f}")
            print(f"l_z: {l_z / H_max:.2f}")
        return l_xm, l_xp, l_y, l_z
