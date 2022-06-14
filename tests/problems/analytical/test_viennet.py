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
"""Tests for the Viennet problem."""
from __future__ import annotations

import pytest
from gemseo_pymoo.problems.analytical.viennet import Viennet
from numpy import array
from numpy.testing import assert_allclose


@pytest.fixture
def viennet() -> Viennet:
    """Create a :class:`gemseo_pymoo.problems.analytical.viennet.Viennet` problem.

    Returns:
        A Viennet instance.
    """
    return Viennet(initial_guess=array([1.0, 1.0]))


def test_obj_jacobian(viennet):
    """Test the jacobian of the Viennet objective function.

    Args:
        viennet: Fixture returning a Viennet `OptimizationProblem`.
    """
    x_dv = array([0.0, 0.0])
    jac = viennet.objective.jac(x_dv)
    assert_allclose(
        [[0.0, 0.0], [3.0 + 2.0 / 27.0, -2.0 - 2.0 / 27.0], [0.0, 0.0]], jac
    )
