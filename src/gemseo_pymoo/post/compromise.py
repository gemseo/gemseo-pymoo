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
"""Compromise points for multi-criteria decision-making."""
from __future__ import annotations

import logging
from typing import Any

from numpy import atleast_2d
from numpy import ndarray
from numpy import vstack
from pymoo.factory import get_decomposition

from gemseo_pymoo.algos.opt_result_mo import Pareto
from gemseo_pymoo.post.scatter_pareto import ScatterPareto

LOGGER = logging.getLogger(__name__)


class Compromise(ScatterPareto):
    """Scatter plot with pareto front and compromise points.

    See `Compromise Programming
    <https://pymoo.org/mcdm/index.html#Compromise-Programming>`_.
    """

    fig_title = "Compromise Points"

    fig_name_prefix = "compromise"

    SCALARIZATION_FUNCTIONS: dict[str, str] = {
        "weighted-sum": "Weighted Sum",
        "tchebi": "Tchebysheff",
        "pbi": "PBI",
        "asf": "Achievement Scalarization Function",
        "aasf": "Augmented Achievement Scalarization Function",
        "perp_dist": "Perpendicular Distance",
    }
    """The alias and the corresponding name of the scalarization functions."""

    def _plot(
        self,
        scalar_name: str = "weighted-sum",
        weights: ndarray | None = None,
        plot_extra: bool = True,
        plot_legend: bool = True,
        plot_arrow: bool = False,
        **scalar_options: Any,
    ) -> None:
        """Scatter plot of the pareto front along with the compromise points.

        The compromise points are calculated using a
        `scalarization function <https://pymoo.org/misc/decomposition.html>`_).

        Args:
            scalar_name: The name of the scalarization function to use.
            weights: The weights for the scalarization function. If None, a normalized
                array is used, e.g. [1./n, 1./n, ..., 1./n] for an optimization problem
                with n-objectives.
            plot_extra: Whether to plot the extra pareto related points,
                i.e. ``utopia``, ``nadir`` and ``anchor`` points.
            plot_legend: Whether to show the legend.
            plot_arrow: Whether to plot arrows connecting the utopia point to
                the compromise points. The arrows are annotated with the ``2-norm`` (
                `Euclidian distance <https://en.wikipedia.org/wiki/Euclidean_distance>`_
                ) of the vector represented by the arrow.
            **scalar_options: The keyword arguments for the scalarization function.

        Raises:
            ValueError: Either if the scalarization function name is unknown or if the
                number of weights does not match the number of objectives.
        """
        if scalar_name not in self.SCALARIZATION_FUNCTIONS:
            raise ValueError(
                "The scalarization function name must be one of the "
                f"following: {self.SCALARIZATION_FUNCTIONS}"
            )

        # Objectives.
        n_obj = self.opt_problem.objective.dim

        # Default weights.
        if weights is None:
            weights = [1.0 / n_obj] * n_obj

        # Ensure correct dimension and type.
        weights = atleast_2d(weights).astype(float)

        # Check weight's dimension.
        if weights.shape[1] != n_obj:
            raise ValueError(
                "You must provide exactly one weight for each objective function!"
            )

        # Create Pareto object.
        pareto = Pareto(self.opt_problem)

        # Initialize decomposition function.
        decomposition = get_decomposition(scalar_name, **scalar_options)

        # Prepare points to plot.
        points = []  # Points' coordinates.
        point_labels = []  # Points' labels.
        for weight in weights:

            # Apply decomposition.
            d_res = decomposition.do(
                pareto.front,
                weight,
                utopian_point=pareto.utopia,
                nadir_point=pareto.anti_utopia,
                **scalar_options,
            )

            # Best value according to the scalarization function.
            d_min = d_res.min()

            # Index where the minimum value is located (at the pareto front).
            d_idx = d_res.argmin()

            # Point's coordinates.
            points.append(pareto.front[d_idx])

            # Point's label.
            float_format = ".2e" if abs(d_min) > 1e3 else ".2f"
            point_labels.append(f"s({weight}) = {d_min:{float_format}}")

        # Concatenate points to plot.
        points = vstack(points)

        # Extra figure options.
        self.fig_title = (
            f"{self.fig_title}\n(s = {self.SCALARIZATION_FUNCTIONS[scalar_name]})"
        )

        # Update name's prefix with current decomposition function's name.
        self.fig_name_prefix += f"_{scalar_name}"

        super()._plot(points, point_labels, plot_extra, plot_legend, plot_arrow)
