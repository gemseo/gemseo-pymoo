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
"""Settings for post-processing."""

from __future__ import annotations

from functools import partial
from typing import Any

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.utils.pydantic import copy_field
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TC002
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveFloat
from pydantic import field_validator
from pymoo.core.decomposition import Decomposition
from pymoo.decomposition.weighted_sum import WeightedSum

from gemseo_pymoo.post.base_pymoo_post_settings import BasePymooPostSettings
from gemseo_pymoo.post.base_pymoo_post_settings import _array_validation_function

copy_field_opt = partial(copy_field, model=BasePostSettings)


class WeightedPostSettings(BasePymooPostSettings):
    """The settings common to Compromise, Petal and Radar gemseo-pymoo post-processing.

    classes.
    """

    decomposition: Any | None = Field(
        default=None,
        description="The instance of the scalarization function to use. "
        "If ``None``, use a weighted sum.",
    )

    weights: NDArrayPydantic | NDArrayPydantic[NDArrayPydantic] | None = Field(
        default=None,
        description="The weights for the scalarization function. If None, a "
        "normalized array is used, e.g. [1./n, 1./n, ..., 1./n] "
        "for an optimization problem with n-objectives.",
    )

    # scalar opts
    theta: NonNegativeFloat = Field(default=0.0, description="")

    beta: NonNegativeFloat = Field(default=0.0, description="")

    rho: NonNegativeFloat = Field(default=0.0, description="")

    normalize_each_objective: bool = Field(
        default=True, description="Whether the objectives should be normalized."
    )

    fig_size: tuple[PositiveFloat, PositiveFloat] = copy_field_opt(
        "fig_size", default=(10, 6)
    )

    @field_validator("weights")
    @classmethod
    def __check_points(
        cls, weights: NDArrayPydantic | NDArrayPydantic[NDArrayPydantic] | None
    ):
        """Check the size of the weights setting arrays."""
        return _array_validation_function(weights)

    @field_validator("decomposition", mode="before")
    @classmethod
    def __check_decomposition(cls, decomposition: Decomposition | None):
        """Check for the type of the decomposition setting."""
        if decomposition is None:
            decomposition = WeightedSum()
        elif not isinstance(decomposition, Decomposition):
            msg = (
                "The scalarization function must be an instance of "
                "pymoo.core.Decomposition."
            )
            raise TypeError(msg)

        return decomposition
