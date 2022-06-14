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
"""Tests for the Chankong and Haimes problem."""
from __future__ import annotations

import pytest
from gemseo_pymoo.problems.analytical.chankong_haimes import ChankongHaimes
from numpy import array
from numpy.testing import assert_allclose


@pytest.fixture
def chankong_haimes() -> ChankongHaimes:
    """Create a :class:`.ChankongHaimes` optimization problem.

    Returns:
        A ChankongHaimes instance.
    """
    return ChankongHaimes(initial_guess=array([1.0, 1.0]))


def test_obj_jacobian(chankong_haimes):
    """Test the jacobian of the Chankong and Haimes objective function.

    Args:
        chankong_haimes: Fixture returning a ChankongHaimes `OptimizationProblem`.
    """
    x_dv = array([0.0, 0.0])

    jac = chankong_haimes.objective.jac(x_dv)
    assert_allclose([[-4.0, -2.0], [9.0, 2.0]], jac)


def test_constraints_jacobian(chankong_haimes):
    """Test the jacobian Chankong and Haimes constraint functions.

    Args:
        chankong_haimes: Fixture returning a ChankongHaimes `OptimizationProblem`.
    """
    x_dv = array([0.0, 0.0])

    jac0 = chankong_haimes.constraints[0].jac(x_dv)
    assert_allclose([[0.0, 0.0]], jac0)

    jac1 = chankong_haimes.constraints[1].jac(x_dv)
    assert_allclose([[1.0, -3.0]], jac1)
