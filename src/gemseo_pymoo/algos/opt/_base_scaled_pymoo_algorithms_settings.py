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
"""Settings for the optimization algorithms."""

from __future__ import annotations

from functools import partial
from typing import Union

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.utils.pydantic import copy_field
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TCH002
from pydantic import Field
from pydantic import PositiveInt
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.selection import Selection
from strenum import StrEnum

from gemseo_pymoo.algos.opt._base_pymoo_settings import BasePymooSettings

EvolutionaryOperatorTypes = Union[Crossover, Mutation, Sampling, Selection]
copy_field_opt = partial(copy_field, model=BaseOptimizerSettings)


class RefDirsNames(StrEnum):
    energy = "energy"
    das_dennis = "das-dennis"
    multi_layer = "multi-layer"
    layer_energy = "layer-energy"


class BaseScaledPymooAlgorithmsSettings(BasePymooSettings):
    """The common parameters for UNSGA3 and NSGA3 pymoo algorithms."""

    n_partitions: int = Field(
        default=20,
        description="The number of gaps between two "
        "consecutive points along an objective axis.",
    )

    scaling_1: float | None = Field(
        default=None, description="The scaling of the first simplex."
    )

    scaling_2: float | None = Field(
        default=None, description="The scaling of the second simplex."
    )

    n_points: PositiveInt | None = Field(
        default=None, description="The number of points on the unit simplex."
    )

    partitions: NDArrayPydantic | None = Field(
        default=None, description="The custom partitions."
    )

    ref_dirs_name: RefDirsNames | None = Field(
        default="", description="The reference directions."
    )
