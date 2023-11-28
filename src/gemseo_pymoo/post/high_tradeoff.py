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
"""High Trade-Off Points for multi-criteria decision-making."""

from __future__ import annotations

import logging
from typing import Any
from typing import ClassVar

from pymoo.mcdm.high_tradeoff import HighTradeoffPoints

from gemseo_pymoo.algos.opt_result_mo import Pareto
from gemseo_pymoo.post.scatter_pareto import ScatterPareto

LOGGER = logging.getLogger(__name__)


class HighTradeOff(ScatterPareto):
    """Scatter plot with pareto front and high trade-off points.

    See High Trade-Off Points
    `here<https://pymoo.org/mcdm/index.html#nb-high-tradeoff>`_.
    """

    fig_title = "High Trade-Off Points"

    fig_name_prefix = "high_tradeoff"

    prop_interest: ClassVar[dict[str, str]] = {
        "color": "navy",
        "alpha": 1.0,
        "s": 30,
        "zorder": 3,
    }

    def _plot(
        self,
        plot_extra: bool = True,
        plot_legend: bool = True,
        plot_arrow: bool = False,
        **high_tradeoff_options: Any,
    ) -> None:
        """Scatter plot of the pareto front along with the high trade-off points.

        Args:
            plot_extra: Whether to plot the extra pareto related points,
                i.e. ``utopia``, ``nadir`` and ``anchor`` points.
            plot_legend: Whether to show the legend.
            plot_arrow: Whether to plot arrows connecting the utopia point to
                the compromise points. The arrows are annotated with the ``2-norm`` (
                `Euclidian distance <https://en.wikipedia.org/wiki/Euclidean_distance>`_
                ) of the vector represented by the arrow.
            **high_tradeoff_options: The keyword arguments for the class
                :class:`pymoo.mcdm.high_tradeoff.HighTradeoffPoints`.
        """
        # Create Pareto object.
        pareto = Pareto(self.opt_problem)

        # Initialize decomposition function.
        decision_making = HighTradeoffPoints(
            ideal=pareto.utopia,
            nadir=pareto.anti_utopia,
            **high_tradeoff_options,
        )

        # Indexes of the High Trade-Off points.
        indexes_dm = decision_making.do(pareto.front)

        super()._plot(
            pareto.front[indexes_dm],
            "High Trade-Offs",
            plot_extra,
            plot_legend,
            plot_arrow,
        )
