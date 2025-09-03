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

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TC002
from numpy import atleast_2d
from numpy import ndarray


class BasePymooPostSettings(BasePostSettings):
    """The settings common to all the gemseo-pymoo post-processing classes."""


def _array_validation_function(
    setting: NDArrayPydantic[float] | None,
) -> NDArrayPydantic[float] | None:
    if isinstance(setting, ndarray):
        if setting.shape[-1] < 2:
            message = (
                f"{setting} setting must be an 1D array of at least 2 items "
                f"or a 2D array where each individual line contains a "
                f"minimum 2 items."
            )
            raise ValueError(message)

        setting = atleast_2d(setting)
    return setting
