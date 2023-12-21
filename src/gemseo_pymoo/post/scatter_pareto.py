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
"""Scatter plot for multi-objective optimization problems."""

from __future__ import annotations

import logging
from math import degrees
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Union

import matplotlib.pyplot as plt
from gemseo.post.opt_post_processor import OptPostProcessor
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from numpy import arctan
from numpy import atleast_2d
from numpy import ndarray
from numpy import vstack
from numpy.linalg import norm as np_norm
from pymoo.visualization.scatter import Scatter as PymooScatter

from gemseo_pymoo.algos.opt_result_mo import Pareto
from gemseo_pymoo.post.core.plot_features import Annotation3D
from gemseo_pymoo.post.core.plot_features import Arrow3D

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)

PlotPropertiesType = dict[str, Union[str, int, float, bool]]


class ScatterPareto(OptPostProcessor):
    """Scatter plot with pareto points and points of interest.

    See `Scatter <https://pymoo.org/visualization/scatter.html>`_.

    Note:
        This post-processor assumes the optimization has converged to a well-defined
        pareto front.
    """

    fig_title: str = "Pareto"
    """The figure's title."""

    fig_name_prefix: str = "scatter"
    """The figure's name prefix."""

    font_size: int = 9
    """The font size for the plot texts."""

    prop_front: ClassVar[PlotPropertiesType] = {
        "color": "blue",
        "alpha": 0.2,
        "s": 10,
        "zorder": 1,
    }
    """The properties for the pareto points."""

    prop_extra: ClassVar[PlotPropertiesType] = {"alpha": 0.8, "s": 30, "zorder": 2}
    """The properties for the extra points."""

    prop_interest: ClassVar[PlotPropertiesType] = {"alpha": 1.0, "s": 30, "zorder": 3}
    """The properties for the points of interest."""

    prop_arrow: ClassVar[PlotPropertiesType] = {
        "color": "black",
        "alpha": 0.5,
        "arrowstyle": "-|>",
        "mutation_scale": 12,
    }
    """The properties for the arrows."""

    prop_annotation: ClassVar[PlotPropertiesType] = {
        "fontsize": font_size - 2,
        "ha": "center",
        "va": "bottom",
        "transform_rotates_text": True,
        "rotation_mode": "anchor",  # Alignment occurs before rotation.
    }
    """The properties for the annotations."""

    def _plot(
        self,
        points: ndarray | None = None,
        point_labels: Sequence[str] = "points",
        plot_extra: bool = True,
        plot_legend: bool = True,
        plot_arrow: bool = False,
        **scatter_options: Any,
    ) -> None:
        """Scatter plot of the pareto front along with the points of interest.

        Args:
            points: The points of interest to be plotted.
                If None, only the pareto front is plot along with extra point
                (depending on ``plot_extra`` value).
            point_labels: The label of the points of interest. If a list is provided,
                it must contain as many labels as the points of interest.
                Moreover, in the last case, each point will have a different color.
            plot_extra: Whether to plot the extra pareto related points,
                i.e. ``utopia``, ``nadir`` and ``anchor`` points.
            plot_legend: Whether to show the legend.
            plot_arrow: Whether to plot arrows connecting the utopia point to
                the compromise points. The arrows are annotated with the ``2-norm`` (
                `Euclidian distance <https://en.wikipedia.org/wiki/Euclidean_distance>`_
                ) of the vector represented by the arrow.
            **scatter_options: The keyword arguments for the class
                :class:`pymoo.visualization.scatter.Scatter`.

        Raises:
            ValueError: Either if the number of objectives is not 2 or 3,
                or if a list of labels is provided but its size
                does not match the number of points.
        """
        # Objectives.
        n_obj = self.opt_problem.objective.dim
        obj_name = self.opt_problem.objective.name

        # Ensure right dimension.
        points = [] if points is None else atleast_2d(points)

        # Check post-processing suitability.
        if not 2 <= n_obj <= 3:
            raise ValueError(
                "This post-processing is only suitable for optimization "
                "problems with 2 or 3 objective functions!"
            )

        # Check labels.
        if not isinstance(point_labels, str) and len(point_labels) != len(points):
            raise ValueError(
                "You must provide either a single label for all points "
                "or one label for each one!"
            )

        # Create Pareto object.
        pareto = Pareto(self.opt_problem)

        # Default plot options.
        plot_options = {
            "figsize": self.DEFAULT_FIG_SIZE,
            "title": self.fig_title,
            "tight_layout": False,
            "legend": (plot_legend, {"fontsize": self.font_size - 2, "loc": "best"}),
            "labels": [f"{obj_name} ({i + 1})" for i in range(n_obj)],  # Axes' labels.
            "close_on_destroy": False,  # Do not close figure when plot is destroyed.
        }

        # Update default options with user choices.
        plot_options.update(**scatter_options)

        # Create plot.
        plot = PymooScatter(**plot_options)

        # Change font family to ensure good alignment.
        plt.rc("font", family="monospace", size=self.font_size)

        # Plot pareto front.
        plot.add(pareto.front, label="pareto front", **self.prop_front)

        # Plot extra pareto related points.
        if plot_extra:
            plot.add(pareto.anchor_front, label="anchor points", **self.prop_extra)

            utopia_label = f"utopia = {pareto.utopia.round(decimals=2)}"
            plot.add(pareto.utopia, label=utopia_label, **self.prop_extra)

            nadir_label = f" nadir = {pareto.anti_utopia.round(decimals=2)}"
            plot.add(pareto.anti_utopia, label=nadir_label, **self.prop_extra)

        # Plot points of interest.
        if len(points) > 0:
            if isinstance(point_labels, str):
                plot.add(points, label=point_labels, **self.prop_interest)
            else:
                for point, label in zip(points, point_labels):
                    plot.add(point, label=label, **self.prop_interest)
        plot.do()

        # Create arrows.
        if plot_arrow:
            for point in points:
                # Arrow vector.
                vect = point - pareto.utopia
                norm = np_norm(vect)

                if n_obj == 2:
                    arr = FancyArrowPatch(pareto.utopia, point, **self.prop_arrow)
                    rot = 90 if vect[0] == 0 else degrees(arctan(vect[1] / vect[0]))
                    annot = Annotation(
                        f"{norm:.2f}",
                        (0.5 * (pareto.utopia + point)),
                        rotation=rot,
                        **self.prop_annotation,
                    )
                else:
                    vect = vstack([pareto.utopia, point])
                    arr = Arrow3D(vect, **self.prop_arrow)
                    annot = Annotation3D(f"{norm:.2f}", vect, **self.prop_annotation)

                # Plot arrow and text in axes.
                plot.ax.add_artist(arr)
                plot.ax.add_artist(annot)

        file_name = f"{self.fig_name_prefix}_{len(self.figures) + 1}"
        self._add_figure(plot.fig, file_name)
