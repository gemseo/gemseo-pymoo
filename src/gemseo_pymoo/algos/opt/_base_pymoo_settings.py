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
from typing import Any
from typing import Union

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.utils.pydantic import copy_field
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.selection import Selection

EvolutionaryOperatorTypes = Union[Crossover, Mutation, Sampling, Selection]
copy_field_opt = partial(copy_field, model=BaseOptimizerSettings)


class BasePymooSettings(BaseOptimizerSettings):
    """The common parameters for all optimization libraries."""

    max_gen: PositiveInt = Field(
        default=10000000,
        description="The maximum number of generations.",
    )

    hv_tol_rel: NonNegativeFloat = Field(
        default=1e-9,
        description="A stop criterion, the relative tolerance on the hypervolume "
        "convergence check. If norm(xk-xk+1)/norm(xk)<= hv_tol_rel: stop.",
    )

    hv_tol_abs: NonNegativeFloat = Field(
        default=1e-9,
        description="hv_tol_abs: A stop criterion, absolute tolerance on the "
        "hypervolume convergence check. "
        "If norm(xk-xk+1)<= hv_tol_abs: stop.",
    )

    n_processes: PositiveInt = Field(
        default=1, description="Number of processes for multiprocess problems."
    )

    stop_crit_n_hv: PositiveInt = Field(
        default=5,
        ge=2,
        description="The number of generations to account for "
        "during the criterion check on the hypervolume indicator.",
    )

    normalize_design_space: bool = Field(
        default=True, description="If True, scale the variables to the range [0, 1]."
    )

    eq_tolerance: NonNegativeFloat = Field(
        default=1e-2, description="The equality tolerance."
    )

    ineq_tolerance: NonNegativeFloat = Field(
        default=1e-4, description="The inequality tolerance."
    )

    pop_size: PositiveInt = Field(default=100, description="The population size.")

    sampling: Any | None = Field(
        default=None,
        description="The sampling process that generates the initial population. "
        "If None, the algorithm's default is used",
    )
    selection: Selection | None = Field(
        default=None,
        description="The mating selection operator. "
        "If None, the algorithm's default is used.",
    )

    mutation: Mutation | None = Field(
        default=None,
        description="The mutation operator. If None, the algorithm's default is used.",
    )

    crossover: Crossover | None = Field(
        default=None,
        description="The crossover operator used to create offsprings. "
        "If None, the algorithm's default is used.",
    )

    seed: PositiveInt = Field(default=1, description="The random seed to be used.")

    eliminate_duplicates: bool = Field(
        default=True,
        description="If True, eliminate duplicates after merging the parent "
        "and the offspring population.",
    )

    n_offsprings: NonNegativeInt | None = Field(
        default=None,
        description="Number of offspring that are created through mating. "
        "If None, it will be set equal to the population size.",
    )

    max_iter: PositiveInt = copy_field_opt("max_iter", default=999)

    ftol_rel: float = copy_field_opt("ftol_rel", default=1e-9)

    ftol_abs: float = copy_field_opt("ftol_abs", default=1e-9)

    xtol_rel: float = copy_field_opt("xtol_rel", default=1e-9)

    xtol_abs: float = copy_field_opt("xtol_abs", default=1e-9)
