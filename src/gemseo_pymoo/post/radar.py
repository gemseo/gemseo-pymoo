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
"""Radar plot."""
from __future__ import annotations

import logging
from typing import Any

from numpy import ndarray

from gemseo_pymoo.post.core.multi_objective_diagram import MultiObjectiveDiagram

LOGGER = logging.getLogger(__name__)


class Radar(MultiObjectiveDiagram):
    """`Radar plots <https://pymoo.org/visualization/radar.html>`_).

    Note:
        This post-processor assumes the optimization has converged to a well-defined
        pareto front.
    """

    def _plot(
        self,
        scalar_name: str,
        weights: ndarray,
        normalize_each_objective: bool = True,
        **scalar_options: Any,
    ) -> None:
        """Plot one radar diagram for each set of weights.

        A `scalarization function <https://pymoo.org/misc/decomposition.html>`_ is used
        to transform the multi-objective functions into a single-objective.

        Args:
            scalar_name: The name of the scalarization function to use.
            weights: The weights for the scalarization function.
            normalize_each_objective: Whether the objectives should be normalized.
            **scalar_options: The keyword arguments for the scalarization function.
        """
        super()._plot(
            "radar",
            scalar_name,
            weights,
            normalize_each_objective=normalize_each_objective,
            **scalar_options,
        )
