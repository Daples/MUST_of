from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Box:
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

    def expand(self, l_xm: float, l_xp: float, l_y: float, l_z: float) -> Box:
        """Create a guidelines box based on the current box."""

        return Box(
            xmin=self.xmin - l_xm,
            xmax=self.xmax + l_xp,
            ymin=self.ymin - l_y,
            ymax=self.ymax + l_y,
            zmin=self.zmin,
            zmax=self.zmax + l_z,
        )
