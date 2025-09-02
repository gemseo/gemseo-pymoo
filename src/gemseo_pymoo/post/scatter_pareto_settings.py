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

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Settings for post-processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TC002
from pydantic import Field
from pydantic import field_validator

from gemseo_pymoo.post.base_pymoo_plot_post_settings import BasePlotPostSettings
from gemseo_pymoo.post.base_pymoo_post_settings import _array_validation_function

if TYPE_CHECKING:
    from collections.abc import Sequence


class ScatterParetoPostSettings(BasePlotPostSettings):
    """The settings for  the Scatter Pareto gemseo-pymoo post-processing class."""

    _TARGET_CLASS_NAME = "ScatterPareto"

    points: NDArrayPydantic[float] | None = Field(
        default=None,
        description="The points of interest to be plotted. If `None`, only the pareto "
        "front is plot along with extra point (depending on "
        "`plot_extra` value).",
    )
    points_labels: Sequence[str] | str = Field(
        default="points",
        description="The label of the points of interest. If a list is provided, "
        "it must contain as many labels as the points of interest. "
        "Moreover, in the last case, each point will have a different "
        "color.",
    )

    @field_validator("points")
    @classmethod
    def __check_points(
        cls, points: NDArrayPydantic[float] | None
    ) -> NDArrayPydantic[float] | None:
        """Check the size of the points setting arrays."""
        return _array_validation_function(points)

    @field_validator("points_labels")
    @classmethod
    def __check_labels_size(
        cls,
        points_labels: Sequence[str] | str,
        points: NDArrayPydantic[float] | None,
    ) -> Sequence[str] | str:
        """Check that the number of labels corresponds to the number of points."""
        if not isinstance(points_labels, str) and len(points_labels) != len(
            points.data["points"]
        ):
            msg = (
                "You must provide either a single label for all points "
                "or one label for each one!"
            )
            raise ValueError(msg)

        return points_labels
