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
"""Petal diagram."""

from __future__ import annotations

from typing import ClassVar

from pymoo.visualization.petal import Petal as PymooPetal

from gemseo_pymoo.post.core.multi_objective_diagram import MultiObjectiveDiagram
from gemseo_pymoo.post.petal_settings import PetalPostSettings


class Petal(MultiObjectiveDiagram):
    """`Petal diagrams <https://pymoo.org/visualization/petal.html>`_).

    Note:
        This post-processor assumes the optimization has converged to a well-defined
        pareto front.
    """

    _TARGET_CLASS_NAME = "Petal"

    Settings: ClassVar[type[PetalPostSettings]] = PetalPostSettings

    def _plot(self, settings: PetalPostSettings) -> None:
        """Plot one petal diagram for each set of weights.

        A `scalarization function <https://pymoo.org/misc/decomposition.html>`_ is used
        to transform the multi-objective functions into a single-objective.
        """
        super()._plot(visualization=PymooPetal, settings=settings)
