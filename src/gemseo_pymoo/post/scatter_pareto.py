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

from math import degrees
from typing import ClassVar
from typing import Union

import matplotlib.pyplot as plt
from gemseo.algos.pareto.pareto_front import ParetoFront
from gemseo.post.base_post import BasePost
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from numpy import arctan
from numpy import vstack
from numpy.linalg import norm as np_norm
from pymoo.visualization.scatter import Scatter as PymooScatter

from gemseo_pymoo.post.core.plot_features import Annotation3D
from gemseo_pymoo.post.core.plot_features import Arrow3D
from gemseo_pymoo.post.scatter_pareto_settings import ScatterParetoPostSettings

PlotPropertiesType = dict[str, Union[str, int, float, bool]]


class ScatterPareto(BasePost[ScatterParetoPostSettings]):
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

    Settings: ClassVar[type[ScatterParetoPostSettings]] = ScatterParetoPostSettings

    def _plot(self, settings: ScatterParetoPostSettings) -> None:
        """Scatter plot of the pareto front along with the points of interest.

        Raises:
            ValueError: Either if the number of objectives is not 2 or 3,
                or if a list of labels is provided but its size
                does not match the number of points.
        """
        # Objectives.
        n_obj = self.optimization_problem.objective.dim
        obj_name = self.optimization_problem.objective.name

        # Check post-processing suitability.
        if not 2 <= n_obj <= 3:
            msg = (
                "This post-processing is only suitable for optimization "
                "problems with 2 or 3 objective functions!"
            )
            raise ValueError(msg)

        # Create Pareto object.
        pareto = ParetoFront.from_optimization_problem(self.optimization_problem)

        # Default plot settings.
        plot_settings = {
            "figsize": settings.fig_size,
            "title": self.fig_title,
            "tight_layout": False,
            "legend": (
                settings.plot_legend,
                {"fontsize": self.font_size - 2, "loc": "best"},
            ),
            "labels": [f"{obj_name} ({i + 1})" for i in range(n_obj)],  # Axes' labels.
            "close_on_destroy": False,  # Do not close figure when plot is destroyed.
        }

        # Update default settings with user choices.
        plot_settings.update(**settings.model_dump())

        # Create plot.
        plot = PymooScatter(**plot_settings)

        # Change font family to ensure good alignment.
        plt.rc("font", family="monospace", size=self.font_size)

        # Plot pareto front.
        plot.add(pareto.f_optima, label="pareto front", **self.prop_front)

        # Plot extra pareto related points.
        if settings.plot_extra:
            plot.add(pareto.f_anchors, label="anchor points", **self.prop_extra)

            utopia_label = f"utopia = {pareto.f_utopia.round(decimals=2)}"
            plot.add(pareto.f_utopia, label=utopia_label, **self.prop_extra)

            nadir_label = f" nadir = {pareto.f_anti_utopia.round(decimals=2)}"
            plot.add(pareto.f_anti_utopia, label=nadir_label, **self.prop_extra)

        # Plot points of interest.
        if settings.points is not None:
            if isinstance(settings.points_labels, str):
                plot.add(
                    settings.points, label=settings.points_labels, **self.prop_interest
                )
            else:
                for point, label in zip(settings.points, settings.points_labels):
                    plot.add(point, label=label, **self.prop_interest)
        plot.do()

        # Create arrows.
        if settings.plot_arrow:
            for point in settings.points:
                # Arrow vector.
                vect = point - pareto.f_utopia
                norm = np_norm(vect)

                if n_obj == 2:
                    arr = FancyArrowPatch(pareto.f_utopia, point, **self.prop_arrow)
                    rot = 90 if vect[0] == 0 else degrees(arctan(vect[1] / vect[0]))
                    annot = Annotation(
                        f"{norm:.2f}",
                        (0.5 * (pareto.f_utopia + point)),
                        rotation=rot,
                        **self.prop_annotation,
                    )
                else:
                    vect = vstack([pareto.f_utopia, point])
                    arr = Arrow3D(vect, **self.prop_arrow)
                    annot = Annotation3D(f"{norm:.2f}", vect, **self.prop_annotation)

                # Plot arrow and text in axes.
                plot.ax.add_artist(arr)
                plot.ax.add_artist(annot)

        file_name = f"{self.fig_name_prefix}_{len(self.figures) + 1}"
        self._add_figure(plot.fig, file_name)
