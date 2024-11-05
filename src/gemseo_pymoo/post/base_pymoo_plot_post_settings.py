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
"""Settings for post-processing."""

from __future__ import annotations

from pydantic import Field

from gemseo_pymoo.post.base_pymoo_post_settings import BasePymooPostSettings


class BasePlotPostSettings(BasePymooPostSettings):
    """The settings common to High Tradeoff, Composition and Scatter Pareto.

    post-processing classes.
    """

    plot_extra: bool = Field(
        default=True,
        description=" Whether to plot the extra pareto related points,"
        "i.e. ``utopia``, ``nadir`` and ``anchor`` points.",
    )

    plot_legend: bool = Field(default=True, description="Whether to show the legend.")

    plot_arrow: bool = Field(
        default=False,
        description="Whether to plot arrows connecting the utopia point to the "
        "compromise points. The arrows are annotated with the "
        "``2-norm`` (`Euclidian distance "
        "<https://en.wikipedia.org/wiki/Euclidean_distance>`_) "
        "of the vector represented by the arrow.",
    )
