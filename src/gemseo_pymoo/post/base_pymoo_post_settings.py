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
from numpy import atleast_2d
from numpy import ndarray


class BasePymooPostSettings(BasePostSettings):
    """The settings common to all the gemseo-pymoo post-processing classes."""


def _array_validation_function(setting_to_validate):
    message = (
        f"{setting_to_validate} setting must be an array of at least 2 items "
        f"or an array of arrays where each individual arrays contains "
        f"minimum 2 items."
    )
    if isinstance(setting_to_validate, ndarray) and all(
        isinstance(item, ndarray) for item in setting_to_validate
    ):
        if all(len(item) < 2 for item in setting_to_validate):
            raise ValueError(message)
        setting_to_validate = atleast_2d(setting_to_validate)
    elif isinstance(setting_to_validate, ndarray):
        if len(setting_to_validate) < 2:
            raise ValueError(message)
        setting_to_validate = atleast_2d(setting_to_validate)
    return setting_to_validate
