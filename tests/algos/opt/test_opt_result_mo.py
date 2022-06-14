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
"""Tests for the multi-objective optimization result."""
from __future__ import annotations

import pytest
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.problems.analytical.power_2 import Power2
from gemseo_pymoo.algos.opt.core.pymoo_problem_adapater import import_hdf
from gemseo_pymoo.algos.opt_result_mo import MultiObjectiveOptimizationResult
from gemseo_pymoo.algos.opt_result_mo import Pareto
from gemseo_pymoo.problems.analytical.chankong_haimes import ChankongHaimes
from gemseo_pymoo.problems.analytical.viennet import Viennet
from numpy import array
from numpy.testing import assert_array_equal
from pandas import DataFrame
from pandas import MultiIndex


@pytest.fixture
def result() -> MultiObjectiveOptimizationResult:
    """The optimization result of the Viennet optimization problem."""
    problem = Viennet()
    res = OptimizersFactory().execute(problem, algo_name="PYMOO_NSGA2", max_iter=700)
    return res


@pytest.fixture
def problem_1obj() -> OptimizationProblem:
    """An optimization problem with 1 objective ready to be post-processed."""
    power2 = Power2()
    power2.constraints = power2.get_ineq_constraints()
    OptimizersFactory().execute(power2, algo_name="PYMOO_NSGA2", max_iter=100)
    return power2


@pytest.fixture
def problem_2obj() -> OptimizationProblem:
    """An optimization problem with 2 objectives ready to be post-processed."""
    problem = ChankongHaimes()
    OptimizersFactory().execute(problem, algo_name="PYMOO_NSGA2", max_iter=100)
    return problem


def test_pareto(problem_1obj, problem_2obj):
    """Test the :class:`Pareto` class.

    Args:
        problem_1obj: Fixture returning the single-objective
            optimization problem to post-process.
        problem_2obj: Fixture returning the multi-objective
            optimization problem to post-process.
    """
    with pytest.raises(
        Exception, match="Single-objective problems have no Pareto Front."
    ):
        Pareto(problem_1obj)

    obj_name = problem_2obj.objective.name
    database = problem_2obj.database
    keys = list(database.keys())
    # Take the objective evaluations out from database except for the two firsts.
    for key in keys[2:]:
        database.get(key).pop(problem_2obj.objective.name)
    pareto = Pareto(problem_2obj)

    # Check problem property.
    assert pareto.problem == problem_2obj

    obj0 = database.get_f_of_x(obj_name, keys[0])
    obj1 = database.get_f_of_x(obj_name, keys[1])
    if len(pareto.front) == 1:
        assert (obj0 in pareto.front) or (obj1 in pareto.front)
    else:
        assert (obj0 in pareto.front) and (obj1 in pareto.front)

    # Check if the anchor points are in database.
    for anchor_point, anchor_obj in zip(pareto.anchor_set, pareto.anchor_front):
        assert_array_equal(
            database.get_f_of_x(problem_2obj.objective.name, anchor_point), anchor_obj
        )

    # Check if the minimum norm point is in database.
    assert pareto.min_norm_f in database.get_func_history(problem_2obj.objective.name)


def test_get_lowest_norm(result):
    """Test the method :meth:`gemseo_pymoo.algos.opt_result_mo.Pareto.get_lowest_norm`.

    Args:
        result: Fixture returning a
            :class:`gemseo_pymoo.algos.opt_result_mo.MultiObjectiveOptimizationResult`.
    """
    # 2 objectives, 3 design variables and 4 points on the pareto front.
    p_front = array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    p_set = array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])

    min_norm_f, min_norm_x, min_norm = result.pareto.get_lowest_norm(
        p_front, p_set, reference=None
    )

    assert_array_equal(min_norm_f, p_front[[1, 2]])
    assert_array_equal(min_norm_x, p_set[[1, 2]])
    assert min_norm == (2**2 + 3**2) ** 0.5

    assert "Pareto available: True" in repr(result)

    with pytest.raises(
        Exception,
        match="does not have the same amount of objectives as the pareto front",
    ):
        result.pareto.get_lowest_norm(p_front, p_set, reference=array([1.0, 1.0, 1.0]))


def test_pretty_table(result):
    """Test the creation of a :class:`gemseo.third_party.prettytable.PrettyTable`.

    Args:
        result: Fixture returning a
            :class:`gemseo_pymoo.algos.opt_result_mo.MultiObjectiveOptimizationResult`.
    """
    # Create DataFrame with multiple index and column levels.
    dframe = DataFrame(dict(a=[1, 2, 3, 4], b=[5, 6, 7, 8]))
    indexes = [("i", "1"), ("i", "2"), ("j", "1"), ("j", "2")]
    dframe.index = MultiIndex.from_tuples(indexes)
    columns = [("c", "a"), ("c", "b")]
    dframe.columns = MultiIndex.from_tuples(columns)

    p_table = str(result.pareto.get_pretty_table_from_df(dframe))
    for val in columns + indexes:
        assert f"{val[0]} ({val[1]})" in p_table

    # Drop multiple indexes.
    dframe.reset_index(level=[0, 1], drop=True, inplace=True)
    p_table = str(result.pareto.get_pretty_table_from_df(dframe))
    for val in range(4):
        assert str(val) in p_table


def test_export_import_optimization_history(tmp_wd, problem_2obj):
    """Test the export and the import of the multi-objective optimization history.

    Args:
        tmp_wd: Fixture to move into a temporary working directory.
        problem_2obj: Fixture returning the multi-objective
            optimization problem to export.
    """
    file_path = tmp_wd / "problem.h5"
    problem_2obj.export_hdf(file_path)

    assert file_path.exists()

    problem_2obj_imported = import_hdf(file_path)

    assert isinstance(problem_2obj_imported.solution, MultiObjectiveOptimizationResult)
    assert isinstance(problem_2obj_imported.solution.pareto, Pareto)
    assert str(problem_2obj_imported.solution) == str(problem_2obj.solution)


def test_export_import_empty_optimization_history(tmp_wd):
    """Test the export and the import of an empty optimization history.

    Args:
        tmp_wd: Fixture to move into a temporary working directory.
    """
    file_path = tmp_wd / "problem.h5"

    problem = Power2()
    problem.export_hdf(file_path)

    assert file_path.exists()

    problem_imported = import_hdf(file_path)

    assert problem_imported.solution is None
    assert str(problem_imported.solution) == str(problem.solution)
