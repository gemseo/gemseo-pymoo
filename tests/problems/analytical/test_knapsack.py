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
"""Tests for the Knapsack problem."""
from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo_pymoo.problems.analytical.knapsack import create_random_knapsack_problem
from gemseo_pymoo.problems.analytical.knapsack import Knapsack
from gemseo_pymoo.problems.analytical.knapsack import MultiObjectiveKnapsack
from numpy import arange
from numpy import array
from numpy import ones
from numpy import zeros
from numpy.testing import assert_array_equal

integer_operators = dict(
    sampling="int_lhs",
    crossover="int_sbx",
    mutation=("int_pm", dict(prob=1.0, eta=3.0)),
)
integer_options = dict(normalize_design_space=False, stop_crit_n_x=99)


@pytest.fixture
def knapsack_max_items() -> Knapsack:
    """Create a :class:`.Knapsack` optimization problem.

    Returns:
        A Knapsack instance constrained by the number of items.
    """
    values = array([55.0, 10.0, 47.0, 5.0, 4.0, 50.0, 8.0, 61.0, 85.0, 87.0])
    weights = array([95.0, 4.0, 60.0, 32.0, 23.0, 72.0, 80.0, 62.0, 65.0, 46.0])
    knapsack = Knapsack(
        values,
        weights,
        # capacity_weight=269.0,
        capacity_items=5,
        initial_guess=ones(10),
    )
    # knapsack.solution = (array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1]), array([-295.0]))
    knapsack.solution = (array([1, 0, 0, 0, 0, 1, 0, 1, 1, 1]), array([-338.0]))
    return knapsack


@pytest.fixture
def mo_knapsack() -> MultiObjectiveKnapsack:
    """Create a :class:`.MultiObjectiveKnapsack` optimization problem.

    Returns:
        A MultiObjectiveKnapsack instance.
    """
    return MultiObjectiveKnapsack(
        array([55.0, 10.0, 47.0, 5.0, 4.0, 50.0, 8.0, 61.0, 85.0, 87.0]),
        array([95.0, 4.0, 60.0, 32.0, 23.0, 72.0, 80.0, 62.0, 65.0, 46.0]),
        capacity_weight=269.0,
        initial_guess=ones(10),
    )


@pytest.mark.parametrize(
    "args, expectation",
    [
        (
            [arange(5), arange(4), ones(5)],
            pytest.raises(
                ValueError,
                match="weights and values must have the same number of elements!",
            ),
        ),
        (
            [arange(5), arange(5), ones(5)],
            pytest.raises(
                ValueError, match="You have to provide at least one type of capacity!"
            ),
        ),
        (
            [arange(5), arange(5), ones(4), True, 100.0],
            pytest.raises(
                ValueError,
                match="items_ub and values must have the same number of elements!",
            ),
        ),
        (
            [arange(5), arange(5), ones(5), True, 100.0, 5, ones(4)],
            pytest.raises(
                ValueError,
                match=r"initial_guess must have \d+ elements!",
            ),
        ),
    ],
)
def test_exceptions(args, expectation):
    """Test the exceptions raised by :class:`.Knapsack`.

    Args:
        args: The arguments for the :class:`.Knapsack`.
        expectation: The expected exception.
    """
    with expectation:
        Knapsack(*args)


def test_warnings(caplog):
    """Test the warnings raised by :class:`.Knapsack`.

    Args:
        caplog: Fixture to access and control log capturing.
    """
    Knapsack(arange(5), arange(5), 2 * ones(5), True, 100.0)
    assert "binary option is ignored" in caplog.text


@pytest.mark.parametrize(
    "args, expectation",
    [
        (
            [0],
            pytest.raises(
                ValueError, match="Number of items must be a positive number!"
            ),
        ),
        (
            [10, 2.0],
            pytest.raises(
                ValueError, match=r"capacity_level must be in the interval \(0, 1\)!"
            ),
        ),
        ([10, 0.5, True, "multi"], does_not_raise()),
        ([10, 0.5, False, "single"], does_not_raise()),
    ],
)
def test_problem_creator(args, expectation):
    """Test the :meth:`.create_random_knapsack_problem` method.

    Args:
        args: The arguments for the :meth:`.create_random_knapsack_problem`.
        expectation: The expected exception.
    """
    with expectation:
        create_random_knapsack_problem(*args)


def test_optimization(knapsack_max_items):
    """Test the knapsack problem with the ``capacity_items`` constraint.

    Args:
        knapsack_max_items: Fixture returning a Knapsack `OptimizationProblem`
            constrained by the number of items.
    """
    x_opt, f_opt = knapsack_max_items.solution

    options = dict(max_iter=800, max_gen=20, **integer_operators, **integer_options)
    res = OptimizersFactory().execute(
        knapsack_max_items, algo_name="PYMOO_NSGA2", **options
    )

    assert_array_equal(x_opt, res.x_opt)
    assert_array_equal(f_opt, res.f_opt)


def test_mo_maximize(mo_knapsack):
    """Test the ``maximize objective`` option.

    Args:
        mo_knapsack: Fixture returning a MultiObjectiveKnapsack `OptimizationProblem`.
    """
    mo_knapsack.change_objective_sign()

    options = dict(max_iter=500, max_gen=20, **integer_operators, **integer_options)
    res = OptimizersFactory().execute(mo_knapsack, algo_name="PYMOO_NSGA2", **options)

    # Known solution (one of the anchor points).
    anchor_x = zeros(10)
    anchor_f = zeros(2)
    assert anchor_x in res.pareto.set
    assert anchor_f in res.pareto.front
