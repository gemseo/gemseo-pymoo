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

from typing import ClassVar

from gemseo.algos.pareto.pareto_front import ParetoFront
from numpy import atleast_2d

from gemseo_pymoo.post.compromise_settings import CompromisePostSettings
from gemseo_pymoo.post.core.decomposition_application import _apply_decomposition
from gemseo_pymoo.post.scatter_pareto import ScatterPareto
from gemseo_pymoo.post.scatter_pareto_settings import ScatterParetoPostSettings


class Compromise(ScatterPareto):
    """Scatter plot with pareto front and compromise points.

    See
    `Compromise Programming <https://pymoo.org/mcdm/index.html#Compromise-Programming>`_.
    """

    fig_title = "Compromise Points"

    fig_name_prefix = "compromise"

    Settings: ClassVar[type[CompromisePostSettings]] = CompromisePostSettings

    def _plot(self, settings: CompromisePostSettings) -> None:
        """Scatter plot of the pareto front along with the compromise points.

        The compromise points are calculated using a
        `scalarization function <https://pymoo.org/misc/decomposition.html>`_).

        Raises:
            TypeError: If the scalarization function is not an instance of
                ``Decomposition``.
            ValueError: If the number of weights does not match the number
                 of objectives.
        """
        # Objectives.
        n_obj = self.optimization_problem.objective.dim

        # Default weights.
        if settings.weights is None:
            settings.weights = [1.0 / n_obj] * n_obj

        settings.weights = atleast_2d(settings.weights).astype(float)
        # Check weight's dimension.
        if settings.weights.shape[1] != n_obj:
            msg = "You must provide exactly one weight for each objective function!"
            raise ValueError(msg)

        # Create Pareto object.
        pareto = ParetoFront.from_optimization_problem(self.optimization_problem)

        # Prepare points to plot.
        points, points_labels = [], []  # Points' coordinates, # Points' labels.

        # apply decomposition
        points, points_labels = _apply_decomposition(
            points, points_labels, pareto, settings
        )

        # Extra figure settings.
        self.fig_title = (
            f"{self.fig_title}\n(s = {settings.decomposition.__class__.__name__})"
        )

        # Update name's prefix with current decomposition function's name.
        self.fig_name_prefix += f"_{settings.decomposition.__class__.__name__}"

        super()._plot(
            ScatterParetoPostSettings(
                points=points,
                points_labels=points_labels,
                plot_extra=settings.plot_extra,
                plot_legend=settings.plot_legend,
                plot_arrow=settings.plot_arrow,
            )
        )
