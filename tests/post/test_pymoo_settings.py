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
#        :author: Gilberto RUIZ JIMENEZ
"""Tests for the pymoo post-processing settings."""

import pytest
from numpy import array

from gemseo_pymoo.post.compromise_settings import CompromisePostSettings
from gemseo_pymoo.post.petal_settings import PetalPostSettings
from gemseo_pymoo.post.radar_settings import RadarPostSettings
from gemseo_pymoo.post.scatter_pareto_settings import ScatterParetoPostSettings


@pytest.mark.parametrize(
    ("settings_model", "setting_name"),
    [
        (
            ScatterParetoPostSettings,
            "points",
        ),
        (RadarPostSettings, "weights"),
        (PetalPostSettings, "weights"),
        (CompromisePostSettings, "weights"),
    ],
)
@pytest.mark.parametrize(
    ("setting_value", "fail"),
    [
        (
            array(
                [[1.0, 2.0], [2.0, 5.0]],
            ),
            False,
        ),
        (
            array(
                [[1.0], [2.0], [3.0]],
            ),
            True,
        ),
    ],
)
def test_array_validation_function(settings_model, setting_name, setting_value, fail):
    """Test the array validator defined in the :class:`.BasePymooPostSettings`."""
    if fail:
        with pytest.raises(
            ValueError,
            match=r"The given value must be an array with at least"
            " 2 items on its last dimension.",
        ):
            settings_model(**{setting_name: setting_value})
    else:
        settings_model(**{setting_name: setting_value})
