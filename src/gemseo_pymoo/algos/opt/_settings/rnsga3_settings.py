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
"""Settings for the PYMOO RNSGA3 algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TC002
from pydantic import Field

from gemseo_pymoo.algos.opt._base_pymoo_settings import BasePymooSettings

if TYPE_CHECKING:
    from pydantic import NonNegativeFloat
    from pydantic import PositiveInt


class RNSGA3Settings(BasePymooSettings):
    """The settings for the PYMOO RNSGA3 algorithm."""

    _TARGET_CLASS_NAME = "PYMOO_RNSGA3"

    mu: NonNegativeFloat = Field(
        default=0.1,
        description="The scaling of the reference lines used "
        "during survival selection.",
    )

    pop_per_ref_point: PositiveInt = Field(
        default=1,
        description="The size of the population used for each reference point.",
    )

    ref_points: NDArrayPydantic | None = Field(
        default=None,
        description="The reference points (Aspiration Points) as a NumPy array "
        "where each row represents a point and each column a variable.",
    )
