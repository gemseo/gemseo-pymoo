# Copyright 2022 Airbus SAS
# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Gabriel Max DE MENDONÇA ABRANTES
"""Features for scatter plots."""

from __future__ import annotations

from math import degrees
from typing import TYPE_CHECKING
from typing import Any

from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d import proj3d
from numpy import arctan
from numpy import ndarray

if TYPE_CHECKING:
    from matplotlib.backend_bases import RendererBase


class Arrow3D(FancyArrowPatch):
    """Arrow patch for 3D plots."""

    _vertices_3d: ndarray
    """The 2D array containing the starting and ending positions of the arrow."""

    def __init__(self, xyzs: ndarray, *args: Any, **kwargs: Any) -> None:
        """Instantiate a 3D arrow.

        Args:
            xyzs: 2D array containing the starting and ending positions of the arrow.
            *args: The arguments for the :class:`~matplotlib.patches.FancyArrowPatch`.
            **kwargs: The keyword arguments
                for the :class:`~matplotlib.patches.FancyArrowPatch`.
        """
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._vertices_3d = xyzs

    def do_3d_projection(self) -> float:
        """Update the plot to ensure the right projection of the arrow.

        Returns:
            The minimum z value between the starting and ending positions of the arrow
                after the 3d projection.
        """
        xyz0, xyz1 = self._vertices_3d
        x0, y0, z0 = proj3d.proj_transform(*xyz0, self.axes.M)
        x1, y1, z1 = proj3d.proj_transform(*xyz1, self.axes.M)
        self.set_positions((x0, y0), (x1, y1))

        return min(z0, z1)


class Annotation3D(Annotation):
    """Annotation for 3D plots."""

    _vertices_3d: ndarray
    """The 2D array containing the starting and ending positions of the vector to be
    annotated."""

    def __init__(self, text: str, xyzs: ndarray, *args: Any, **kwargs: Any) -> None:
        """Instantiate a 3D annotation.

        Args:
            text: The annotation text.
            xyzs: 2D array containing the starting and ending positions of the vector to
                be annotated.
            *args: The arguments for the :class:`~matplotlib.text.Annotation`.
            **kwargs: The keyword arguments
                for the :class:`~matplotlib.text.Annotation`.
        """
        Annotation.__init__(self, text, (0, 0), *args, **kwargs)
        self._vertices_3d = xyzs

    def draw(self, renderer: RendererBase) -> None:
        """Update the plot to ensure the right projection of the text.

        Args:
            renderer: The object currently handling the drawing operations.
        """
        xyz0, xyz1 = self._vertices_3d
        x0, y0, _ = proj3d.proj_transform(*xyz0, self.axes.M)
        x1, y1, _ = proj3d.proj_transform(*xyz1, self.axes.M)
        self.set(
            position=(0.5 * (x1 + x0), 0.5 * (y0 + y1)),
            rotation=90 if x0 == x1 else degrees(arctan((y1 - y0) / (x1 - x0))),
        )
        Annotation.draw(self, renderer)
