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
"""Multi-objective diagram base class."""

from __future__ import annotations

from math import ceil
from math import sqrt
from typing import ClassVar

import matplotlib.pyplot as plt
from gemseo.algos.pareto.pareto_front import ParetoFront
from gemseo.post.base_post import BasePost
from gemseo.third_party.prettytable.prettytable import PrettyTable
from matplotlib.gridspec import GridSpec
from numpy.core.shape_base import atleast_2d
from pymoo.core.plot import Plot  # noqa: TC002
from pymoo.visualization.radar import Radar

from gemseo_pymoo.post.base_weighted_pymoo_post_algorithms_settings import (
    WeightedPostSettings,
)
from gemseo_pymoo.post.core.decomposition_application import _apply_decomposition


class MultiObjectiveDiagram(BasePost[WeightedPostSettings]):
    """Base class for post-processing of multi-objective problems."""

    font_size: int = 9
    """The font size for the plot texts."""

    Settings: ClassVar[type[WeightedPostSettings]] = WeightedPostSettings

    def _plot(self, visualization: type[Plot], settings: WeightedPostSettings) -> None:
        """Plot a multi-objective diagram for each set of weights.

        A `scalarization function <https://pymoo.org/misc/decomposition.html>`_ is used
        to transform the multi-objective functions into a single-objective.

        Raises:
            TypeError: If the scalarization function is not an instance of
                ``Decomposition``.
            ValueError: If the number of weights does not match the number
                of objectives, or if the diagram ``radar`` is used for problems with
                less than 3 objectives.
        """
        # Objectives.
        n_obj = self.optimization_problem.objective.dim
        obj_name = self.optimization_problem.objective.name

        settings.weights = atleast_2d(settings.weights).astype(float)
        # Check weight's dimension.
        if settings.weights.shape[1] != n_obj:
            msg = "You must provide exactly one weight for each objective function!"
            raise ValueError(msg)

        # Check post-processing suitability.
        if visualization == Radar and n_obj < 3:
            msg = (
                "The Radar post-processing is only suitable for optimization "
                "problems with at least 3 objective functions!"
            )
            raise ValueError(msg)

        # Create Pareto object.
        pareto = ParetoFront.from_optimization_problem(self.optimization_problem)

        # Prepare points to plot.
        points, title = [], []

        points, title = _apply_decomposition(points, title, pareto, settings)

        # Split in different rows depending on the number of points.
        n_plots = len(points)
        n_cols = ceil(sqrt(n_plots))
        n_rows = ceil(n_plots / n_cols)

        # Go around for the cases with empty subplots.
        while len(title) < n_rows * n_cols:
            title.append([""])

        # Create plot.
        plot = visualization(
            bounds=[pareto.f_utopia, pareto.f_anti_utopia],
            figsize=settings.fig_size,
            title=title,
            tight_layout=False,
            labels=[f"$obj_{i + 1}$" for i in range(n_obj)],
            normalize_each_objective=settings.normalize_each_objective,
            close_on_destroy=False,  # Do not close figure when plot is destroyed.
        )

        # Ensure good alignment.
        plt.rc("font", family="monospace", size=self.font_size)

        # Plot points.
        for row in range(n_rows):
            plot.add(points[row * n_cols : (row + 1) * n_cols])
        plot.do()

        # Adjust subplots.
        gs0 = GridSpec(1, 2, figure=plot.fig, width_ratios=[1, 6])
        gs01 = gs0[1].subgridspec(n_rows, n_cols, wspace=0.2, hspace=0.5)

        for i, axis in enumerate(plot.fig.axes):
            axis.set_position(gs01[i].get_position(plot.fig))
            axis.set_subplotspec(gs01[i])  # Only necessary if using a tight layout.

            # Hide empty axes.
            if not axis.collections:
                axis.set_visible(False)
            else:
                # Update labels manually to include objectives' value.
                for j, val in enumerate(points[i]):
                    float_format = ".2e" if abs(val) >= 1e3 else ".2f"
                    new_text = axis.texts[j].get_text() + f" = {val:{float_format}}"
                    axis.texts[j].set_text(new_text)

            # Adjust label size.
            axis.title.set_size(self.font_size - 1)

        # Prepare text with bounds.
        p_table = PrettyTable()
        p_table.title = "Bounds"
        p_table.add_column(obj_name, range(1, n_obj + 1), align="r")
        p_table.add_column("utopia", pareto.f_utopia.round(decimals=2), align="r")
        p_table.add_column("nadir", pareto.f_anti_utopia.round(decimals=2), align="r")

        # Plot text in a dedicated subplot.
        ax_text = plot.fig.add_subplot(gs0[0])
        ax_text.text(
            0.1,
            0.5,
            str(p_table),
            va="center",
            ha="center",
            fontfamily="monospace",
            fontsize=self.font_size + 1,
        )
        ax_text.axis("off")

        # Set figure title with the scalarization function's name.
        plot.fig.suptitle(f"s = {settings.decomposition.__class__.__name__}")

        self._add_figure(
            plot.fig,
            file_name=f"{visualization.__class__.__name__}_"
            f"{settings.decomposition.__class__.__name__}_{len(self.figures) + 1}",
        )
