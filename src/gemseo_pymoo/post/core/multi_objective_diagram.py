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

import logging
from math import ceil
from math import sqrt
from typing import Any

import matplotlib.pyplot as plt
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.third_party.prettytable import PrettyTable
from matplotlib.gridspec import GridSpec
from numpy import atleast_2d
from numpy import ndarray
from numpy import vstack
from pymoo.factory import get_decomposition
from pymoo.factory import get_visualization

from gemseo_pymoo.algos.opt_result_mo import Pareto

LOGGER = logging.getLogger(__name__)


class MultiObjectiveDiagram(OptPostProcessor):
    """Base class for post-processing of multi-objective problems."""

    DEFAULT_FIG_SIZE = (10, 6)
    """The default width and height of the figure, in inches."""

    font_size: int = 9
    """The font size for the plot texts."""

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
        diagram: str,
        scalar_name: str,
        weights: ndarray,
        normalize_each_objective: bool = True,
        **scalar_options: Any,
    ) -> None:
        """Plot a multi-objective diagram for each set of weights.

        A `scalarization function <https://pymoo.org/misc/decomposition.html>`_ is used
        to transform the multi-objective functions into a single-objective.

        Args:
            diagram: The type of the diagram to be created.
            scalar_name: The name of the scalarization function to use.
            weights: The weights for the scalarization function.
            normalize_each_objective: Whether the objectives should be normalized.
            **scalar_options: The keyword arguments for the scalarization function.

        Raises:
            ValueError: Either if the scalarization function name is unknown, if the
                number of weights does not match the number of objectives, or if the
                diagram ``radar`` is used for problems with less than 3 objectives.
        """
        if scalar_name not in self.SCALARIZATION_FUNCTIONS:
            raise ValueError(
                "The scalarization function name must be one of the "
                f"following: {self.SCALARIZATION_FUNCTIONS}"
            )

        # Ensure correct dimension and type.
        weights = atleast_2d(weights).astype(float)

        # Objectives.
        n_obj = self.opt_problem.objective.dim
        obj_name = self.opt_problem.objective.name

        # Check weight's dimension.
        if weights.shape[1] != n_obj:
            raise ValueError(
                "You must provide exactly one weight for each objective function!"
            )

        # Check post-processing suitability.
        if diagram == "radar" and n_obj < 3:
            raise ValueError(
                "The Radar post-processing is only suitable for optimization "
                "problems with at least 3 objective functions!"
            )

        # Create Pareto object.
        pareto = Pareto(self.opt_problem)

        # Initialize decomposition function.
        decomp = get_decomposition(scalar_name, **scalar_options)

        # Prepare points to plot.
        points, title = [], []
        for weight in weights:

            # Apply decomposition.
            d_res = decomp.do(
                pareto.front,
                weight,
                utopian_point=pareto.utopia,
                nadir_point=pareto.anti_utopia,
                **scalar_options,
            )
            d_min = d_res.min()
            d_idx = d_res.argmin()

            points.append(pareto.front[d_idx])

            float_format = ".2e" if abs(d_min) >= 1e3 else ".2f"
            title.append(f"s({weight}) = {d_min:{float_format}}")

        # Add anchor points (to test coherence of bounds).
        # points.extend([anchor for anchor in pareto.anchor_front])
        # title.extend([f"Anchor ({i + 1})" for i in range(len(pareto.anchor_front))])

        # Concatenate points to plot.
        points = vstack(points)

        # Split in different rows depending on the number of points.
        n_plots = len(points)
        n_cols = ceil(sqrt(n_plots))
        n_rows = ceil(n_plots / n_cols)

        # Go around for the cases with empty subplots.
        while len(title) < n_rows * n_cols:
            title.append([""])

        # Create plot.
        plot = get_visualization(
            diagram,
            bounds=[pareto.utopia, pareto.anti_utopia],
            figsize=self.DEFAULT_FIG_SIZE,
            title=title,
            tight_layout=False,
            labels=[f"$obj_{i + 1}$" for i in range(n_obj)],
            normalize_each_objective=normalize_each_objective,
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
        p_table.add_column("utopia", pareto.utopia.round(decimals=2), align="r")
        p_table.add_column("nadir", pareto.anti_utopia.round(decimals=2), align="r")

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
        plot.fig.suptitle(f"s = {self.SCALARIZATION_FUNCTIONS[scalar_name]}")

        self._add_figure(
            plot.fig, file_name=f"{diagram}_{scalar_name}_{len(self.figures) + 1}"
        )
