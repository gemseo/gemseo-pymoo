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
"""Method to apply decomposition to the different post options."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import vstack

if TYPE_CHECKING:
    from gemseo.algos.pareto.pareto_front import ParetoFront

    from gemseo_pymoo.post.base_pymoo_post_settings import BasePymooPostSettings


def _apply_decomposition(
    points: list,
    names: list,
    pareto: type[ParetoFront],
    settings: type[BasePymooPostSettings],
):
    settings_ = settings.model_dump()
    settings_.pop("weights")
    for weight in settings.weights:
        # Apply decomposition.
        d_res = settings.decomposition.do(
            pareto.f_optima,
            weight,
            utopian_point=pareto.f_utopia,
            nadir_point=pareto.f_anti_utopia,
            **settings_,
        )

        # Best value according to the scalarization function.
        d_min = d_res.min()

        # Index where the minimum value is located (at the pareto front).
        d_idx = d_res.argmin()

        # Point's coordinates.
        points.append(pareto.f_optima[d_idx])

        # Point's label.
        float_format = ".2e" if abs(d_min) > 1e3 else ".2f"
        names.append(f"s({weight}) = {d_min:{float_format}}")

    # For multi objective diagram
    # Add anchor points (to test coherence of bounds).
    # points.extend([anchor for anchor in pareto.anchor_front])
    # title.extend([f"Anchor ({i + 1})" for i in range(len(pareto.anchor_front))])

    # Concatenate points to plot.
    points = vstack(points)

    return points, names
