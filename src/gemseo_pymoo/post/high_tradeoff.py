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

from typing import ClassVar

from gemseo.algos.pareto.pareto_front import ParetoFront
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints

from gemseo_pymoo.post.high_tradeoff_settings import HighTradeOffPostSettings
from gemseo_pymoo.post.scatter_pareto import ScatterPareto
from gemseo_pymoo.post.scatter_pareto_settings import ScatterParetoPostSettings


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

    Settings: ClassVar[type[HighTradeOffPostSettings]] = HighTradeOffPostSettings

    def _plot(self, settings: HighTradeOffPostSettings) -> None:
        """Scatter plot of the pareto front along with the high trade-off points."""
        # Create Pareto object.
        pareto = ParetoFront.from_optimization_problem(self.optimization_problem)

        settings_ = settings.model_dump()
        # Initialize decomposition function.
        decision_making = HighTradeoffPoints(
            ideal=pareto.f_utopia,
            nadir=pareto.f_anti_utopia,
            **settings_,
        )

        # Indexes of the High Trade-Off points.
        indexes_dm = decision_making.do(pareto.f_optima)

        super()._plot(
            settings=ScatterParetoPostSettings(
                points=pareto.f_optima[indexes_dm],
                points_labels="High Trade-Offs",
                plot_extra=settings.plot_extra,
                plot_legend=settings.plot_legend,
                plot_arrow=settings.plot_arrow,
            ),
        )
