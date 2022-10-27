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
#                 François Gallard
#                 Lluis ARMENGOL GARCIA
#                 Luca SARTORI
"""Tests for the Pymoo library wrapper."""
from __future__ import annotations

import logging
from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.problems.analytical.binh_korn import BinhKorn
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo_pymoo.algos.opt.core.pymoo_problem_adapater import get_gemseo_opt_problem
from gemseo_pymoo.problems.analytical.chankong_haimes import ChankongHaimes
from gemseo_pymoo.problems.analytical.knapsack import MultiObjectiveKnapsack
from gemseo_pymoo.problems.analytical.viennet import Viennet
from numpy import array
from numpy import hstack as np_hstack
from numpy import min as np_min
from numpy import ndarray
from numpy.testing import assert_allclose
from pymoo.core.problem import Problem
from pymoo.factory import get_mutation
from pymoo.factory import get_sampling

tolerances = dict(ftol_rel=0.0, ftol_abs=0.0, xtol_rel=0.0, xtol_abs=0.0)
integer_options = dict(normalize_design_space=False, stop_crit_n_x=99)
integer_operators = dict(
    sampling="int_lhs",
    crossover="int_sbx",
    mutation=("int_pm", dict(prob=1.0, eta=3.0)),
)
mixed_operators = dict(
    sampling=dict(integer="int_random", float=dict(custom=get_sampling("real_random"))),
    crossover=dict(
        integer=("int_sbx", dict(prob=1.0, eta=3.0)),
        float=("real_sbx", dict(prob=1.0, eta=3.0)),
    ),
    mutation=dict(int=("int_pm", dict(eta=3.0)), float=("real_pm", dict(eta=3.0))),
)


class MixedVariablesProblem(Problem):
    """Very simple single-objective, constrained MIP problem."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__(n_var=2, n_obj=1, n_constr=1, xl=0, xu=10)

    def _evaluate(
        self, x: ndarray, out: dict[str, ndarray], *args: Any, **kwargs: Any
    ) -> None:
        """Evaluate functions for the given input vector ``x``."""
        out["F"] = -np_min(x * [3, 1], axis=1)
        out["G"] = x[:, 0] + x[:, 1] - 10


class DummyMutation:
    """Dummy mutation operator."""


@pytest.fixture
def pow2_ineq() -> OptimizationProblem:
    """Create a :class:`.Power2` problem with only the inequality constraints.

    Returns:
        A :class:`.Power2` instance.
    """
    power2 = Power2()
    power2.constraints = power2.get_ineq_constraints()

    x_opt = array([0.5 ** (1.0 / 3.0), 0.5 ** (1.0 / 3.0), 0.0])
    f_opt = Power2().pow2(x_opt)
    power2.solution = (x_opt, f_opt)

    return power2


@pytest.fixture
def pow2_unconstrained() -> OptimizationProblem:
    """Create an unconstrained :class:`.Power2` problem.

    Returns:
        A :class:`.Power2` instance.
    """
    power2 = Power2()
    power2.constraints = []

    x_opt = array([0.0, 0.0, 0.0])
    f_opt = Power2().pow2(x_opt)
    power2.solution = (x_opt, f_opt)

    return power2


@pytest.fixture
def pow2_ineq_int() -> OptimizationProblem:
    """Create a Power2 problem with integer variables and only inequality constraints.

    Returns:
        A :class:`.Power2` instance.
    """
    power2 = Power2()
    power2.constraints = power2.get_ineq_constraints()
    power2.design_space.variables_types["x"] = array(["integer"] * 3)

    x_opt = array([1, 1, 0])
    f_opt = Power2().pow2(x_opt)
    power2.solution = (x_opt, f_opt)

    return power2


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
        capacity_items=10,
    )


@pytest.fixture
def simple_mip_problem() -> OptimizationProblem:
    """Create a very simple MIP problem for test purposes.

    Returns:
        A :class:`.MixedVariablesProblem` instance.
    """
    gemseo_problem = get_gemseo_opt_problem(
        MixedVariablesProblem(), mask=["integer", "float"]
    )

    x_opt = array([3, 7.0])
    f_opt = -7.0
    gemseo_problem.solution = x_opt, f_opt

    return gemseo_problem


@pytest.fixture
def opt_factory() -> OptimizersFactory:
    """Create an optimizer factory instance.

    Returns:
        A :class:`.OptimizersFactory` instance.
    """
    return OptimizersFactory()


@pytest.mark.parametrize(
    "algo_name",
    ["PYMOO_NSGA2", "PYMOO_NSGA3", "PYMOO_UNSGA3", "PYMOO_RNSGA3", "PYMOO_GA"],
)
def test_operators_jason_schema(opt_factory, algo_name):
    """Check JSON grammars.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        algo_name: The name of the optimization algorithm.
    """
    options = dict(
        max_iter=1,
        selection=None,
        sampling="s",
        crossover=dict(integer=("string", dict(a=1)), float=("string", dict(b=2))),
        mutation=dict(
            int=dict(custom=get_mutation("int_pm")),
            float=dict(custom=get_mutation("real_pm")),
        ),
    )

    if algo_name == "PYMOO_NSGA2":
        options["pop_size"] = 20
    elif algo_name in ["PYMOO_NSGA3", "PYMOO_UNSGA3"]:
        options["ref_dirs_name"] = "energy"
    elif algo_name == "PYMOO_RNSGA3":
        options["ref_points"] = array([[1.0], [2.0]])

    lib = opt_factory.create(algo_name)
    opt_grammar = lib.init_options_grammar(algo_name)
    try:
        opt_grammar.validate(options, raise_exception=True)
    except InvalidDataException as exception:
        pytest.fail(exception)


@pytest.mark.parametrize(
    "pymoo_problem, expectation",
    [
        (
            Rosenbrock(),
            pytest.raises(Exception, match=f"Problem must be an instance of {Problem}"),
        ),
        ("rosenbrock", does_not_raise()),
    ],
)
def test_get_gemseo_opt_problem(opt_factory, pymoo_problem, expectation):
    """Test the problem conversion :class:`.Problem` -> :class:`.OptimizationProblem`.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        pymoo_problem: The Pymoo optimization problem to be converted.
        expectation: The expected exception to be raised.
    """
    with expectation:
        problem = get_gemseo_opt_problem(pymoo_problem)
        res = opt_factory.execute(problem, algo_name="L-BFGS-B")
        assert abs(res.f_opt) < 1e-5


@pytest.mark.parametrize(
    "options",
    [
        dict(algo_name="PYMOO_GA", pop_size=100),
        dict(algo_name="PYMOO_NSGA2", pop_size=50),
    ],
)
@pytest.mark.parametrize(
    "problem_class, x_opt, f_opt",
    [
        (Power2, array([0.5 ** (1.0 / 3.0), 0.5 ** (1.0 / 3.0), 0]), 1.26),
        (Rosenbrock, array([1, 1]), 0),
    ],
)
def test_so(opt_factory, options, problem_class, x_opt, f_opt):
    """Test the optimization of single-objective problems.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        options: The options for the optimization execution.
        problem_class: The optimization problem class.
        x_opt: The design variables values at the optimum point.
        f_opt: The objective value at the optimum point.
    """
    problem = problem_class()

    # Only inequality constraints are considered.
    problem.constraints = problem.get_ineq_constraints()

    options = dict(stop_crit_n_hv=999, **tolerances, **options)
    res = opt_factory.execute(problem, **options)

    assert_allclose(x_opt, res.x_opt, atol=1e-1)
    assert abs(f_opt - res.f_opt) < 1e-1


@pytest.mark.parametrize(
    "options",
    [
        dict(algo_name="PYMOO_GA", pop_size=100),
        dict(algo_name="PYMOO_NSGA2", pop_size=50),
        dict(algo_name="PYMOO_NSGA3", ref_dirs_name="das-dennis", n_partitions=10),
        dict(algo_name="PYMOO_UNSGA3", ref_dirs_name="das-dennis", n_points=20),
        dict(algo_name="PYMOO_RNSGA3", ref_points=array([[1.0], [5.0]])),
    ],
)
def test_so_hypervolume(opt_factory, pow2_ineq, options, caplog):
    """Test the hypervolume convergence of a single-objective problem.

    The maximum number of iterations is set to a very high number,
    so that the problem is expected to converge according to hypervolume criterion.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        pow2_ineq: Fixture returning the problem to be optimized.
        options: The options for the optimization execution.
        caplog: Fixture to access and control log capturing.
    """
    x_opt, f_opt = pow2_ineq.solution

    options = dict(max_iter=1000000, stop_crit_n_hv=8, **tolerances, **options)
    res = opt_factory.execute(pow2_ineq, **options)

    assert_allclose(x_opt, res.x_opt, atol=1e-1)
    assert abs(f_opt - res.f_opt) < 1e-1

    assert "successive iterates of the hypervolume indicator are closer" in caplog.text


@pytest.mark.parametrize(
    "options",
    [
        dict(algo_name="PYMOO_GA", pop_size=2**10),
        dict(algo_name="PYMOO_NSGA2", pop_size=2**10),
        dict(algo_name="PYMOO_NSGA3", ref_dirs_name="energy", n_points=10),
        dict(algo_name="PYMOO_UNSGA3", ref_dirs_name="energy", n_points=10),
        dict(algo_name="PYMOO_RNSGA3", ref_points=array([[1.0], [5.0]])),
    ],
)
@pytest.mark.parametrize(
    "problem_class, args, kwargs, x_opt, f_opt",
    [
        (Power2, [], {}, array([1, 1, 0]), 2),
        (Rosenbrock, [], {}, array([1, 1]), 0),
    ],
)
def test_so_integer(opt_factory, options, problem_class, args, kwargs, x_opt, f_opt):
    """Test the optimization of single-objective problems with integer variables.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        options: The options for the optimization execution.
        problem_class: The optimization problem class.
        args: The arguments for the optimization problem class.
        kwargs: The keyword arguments for the optimization problem class.
        x_opt: The design variables values at the optimum point.
        f_opt: The objective value at the optimum point.
    """
    problem = problem_class(*args, **kwargs)
    ds_dim = problem.design_space.dimension
    problem.design_space.variables_types["x"] = array(["integer"] * ds_dim)

    # Only inequality constraints are considered.
    problem.constraints = problem.get_ineq_constraints()

    options = dict(
        max_iter=2**11,
        stop_crit_n_hv=999,
        **tolerances,
        **integer_operators,
        **integer_options,
        **options,
    )
    res = opt_factory.execute(problem, **options)

    assert_allclose(x_opt, res.x_opt, atol=1e-1)
    assert abs(f_opt - res.f_opt) < 1e-1


@pytest.mark.parametrize(
    "options",
    [
        dict(algo_name="PYMOO_GA"),
        dict(algo_name="PYMOO_NSGA2"),
        dict(algo_name="PYMOO_NSGA3", ref_dirs_name="das-dennis", n_partitions=20),
        dict(algo_name="PYMOO_UNSGA3", ref_dirs_name="das-dennis", n_partitions=20),
        dict(algo_name="PYMOO_RNSGA3", mu=0.1, ref_points=array([[1.0], [10.0]])),
    ],
)
def test_so_mixed_variables(opt_factory, simple_mip_problem, options):
    """Test the optimization of single-objective problems with mixed variables.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        simple_mip_problem: Fixture returning the problem to be optimized.
        options: The options for the optimization execution.
    """
    mip_x_opt, mip_f_opt = simple_mip_problem.solution

    options = dict(
        max_iter=500,
        pop_size=20,
        stop_crit_n_hv=999,
        **tolerances,
        **mixed_operators,
        **integer_options,
        **options,
    )
    res = opt_factory.execute(simple_mip_problem, **options)

    assert_allclose(res.x_opt, mip_x_opt, atol=1e-1)
    assert abs(mip_f_opt - res.f_opt) < 1e-1


@pytest.mark.parametrize(
    "ref_dirs_options",
    [
        dict(ref_dirs_name="energy", n_points=90),
        dict(ref_dirs_name="das-dennis", n_partitions=20),
        dict(ref_dirs_name="multi-layer", n_partitions=5, scaling_1=1.0, scaling_2=0.5),
        dict(ref_dirs_name="layer-energy", partitions=array([3])),
    ],
)
@pytest.mark.parametrize("algo_name", ["PYMOO_NSGA3", "PYMOO_UNSGA3"])
def test_ref_directions(opt_factory, pow2_ineq, ref_dirs_options, algo_name):
    """Test the different reference directions.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        pow2_ineq: Fixture returning the problem to be optimized.
        ref_dirs_options: The reference direction options.
        algo_name: The name of the optimization algorithm.
    """
    x_opt, f_opt = pow2_ineq.solution

    options = dict(
        max_iter=200, pop_size=20, stop_crit_n_hv=999, **tolerances, **ref_dirs_options
    )
    res = opt_factory.execute(pow2_ineq, algo_name=algo_name, **options)

    assert_allclose(res.x_opt, x_opt, atol=1e-1)
    assert abs(f_opt - res.f_opt) < 1e-1


@pytest.mark.parametrize(
    "options",
    [
        dict(algo_name="PYMOO_NSGA2"),
        dict(algo_name="PYMOO_NSGA3", ref_dirs_name="das-dennis", n_partitions=10),
        dict(algo_name="PYMOO_UNSGA3", ref_dirs_name="das-dennis", n_partitions=10),
        dict(algo_name="PYMOO_RNSGA3", mu=0.5, ref_points_=array([[1.0], [1.0]])),
    ],
)
@pytest.mark.parametrize(
    "problem_class, min_norm_x, min_norm_threshold, atol",
    [
        (ChankongHaimes, array([[-2.6, 10.6]]), 146.5, 5e-1),
        (Viennet, array([[-0.46, 0.32]]), 0.7, 1e-1),
        # (BinhKorn, array([[1.34, 1.33]]), 28, 1e-1),
    ],
)
def test_mo(opt_factory, options, problem_class, min_norm_x, min_norm_threshold, atol):
    """Test the optimization of multi-objectives problems.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        options: The options for the optimization execution.
        problem_class: The optimization problem class.
        min_norm_x: The design variables of the pareto points with the minimum norm.
        min_norm_threshold: The threshold to compare the pareto point with
            the minimum norm.
        atol: The absolute threshold tolerance to compare the points.
    """
    # Instantiate new problem for each test.
    problem = problem_class()

    # Adjust reference points dimensionality according to number of objectives.
    if options["algo_name"] == "PYMOO_RNSGA3":
        n_obj = 2 if problem_class == BinhKorn else problem.objective.dim
        options.update(dict(ref_points=np_hstack([options["ref_points_"]] * n_obj)))

    options = dict(max_iter=700, **tolerances, **options)
    res = opt_factory.execute(problem, **options)

    assert_allclose(res.pareto.min_norm_x, min_norm_x, atol=atol)

    # Verify minimum norm value instead of the compromise function values.
    assert res.pareto.min_norm < min_norm_threshold


def test_mo_integer(opt_factory, mo_knapsack):
    """Test the optimization of a multi-objective problems with integer variables.

    This test allows Pymoo to handle the termination criterion.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        mo_knapsack: Fixture returning the problem to be optimized.
    """
    options = dict(
        max_iter=800,
        pop_size=100,
        stop_crit_n_hv=10,
        **tolerances,
        **integer_operators,
        **integer_options,
    )

    # Instantiate library.
    algo_name = "PYMOO_NSGA2"
    lib = opt_factory.create(algo_name)

    # Manually change the maximum number of generations allowed for Pymoo.
    lib.pymoo_n_gen = 20

    res = lib.execute(mo_knapsack, algo_name=algo_name, **options)

    # Known solution (one of the anchor points).
    anchor_x = array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    anchor_f = array([-295.0, 6])
    assert anchor_x in res.pareto.set
    assert anchor_f in res.pareto.front

    # Best compromise.
    comp_x = array([0, 1, 0, 0, 0, 1, 0, 1, 1, 1])
    comp_f = array([-293.0, 5])
    assert comp_x in res.pareto.set
    assert comp_f in res.pareto.front


@pytest.mark.parametrize("normalize", [True, False])
def test_multiprocessing_constrained(opt_factory, pow2_ineq, normalize):
    """Test the multiprocessing option for a constrained problem.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        pow2_ineq: Fixture returning the problem to be optimized.
        normalize: Whether to normalize the design space.
    """
    x_opt, f_opt = pow2_ineq.solution

    options = dict(
        max_iter=600, pop_size=200, n_processes=2, normalize_design_space=normalize
    )
    res = opt_factory.execute(pow2_ineq, algo_name="PYMOO_NSGA2", **options)

    assert_allclose(res.x_opt, x_opt, atol=1e-1)
    assert abs(f_opt - res.f_opt) < 1e-1


def test_multiprocessing_unconstrained(opt_factory, pow2_unconstrained):
    """Test the multiprocessing option for an unconstrained problem.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        pow2_unconstrained: Fixture returning the problem to be optimized.
    """
    x_opt, f_opt = pow2_unconstrained.solution

    options = dict(max_iter=800, pop_size=200, n_processes=2, stop_crit_n_x=999)
    res = opt_factory.execute(pow2_unconstrained, algo_name="PYMOO_NSGA2", **options)

    assert_allclose(res.x_opt, x_opt, atol=1e-1)
    assert abs(f_opt - res.f_opt) < 1e-1


@pytest.mark.parametrize(
    "problem, options, expectation",
    [
        (
            Viennet(),
            dict(algo_name="PYMOO_GA"),
            pytest.raises(ValueError, match="can not handle multiple objectives."),
        ),
        (
            Rosenbrock(l_b=0, u_b=0),
            dict(
                algo_name="PYMOO_GA", mutation="real_pm", normalize_design_space=False
            ),
            pytest.raises(
                ValueError,
                match="PolynomialMutation cannot handle equal lower and upper bounds!",
            ),
        ),
        (
            get_gemseo_opt_problem(MixedVariablesProblem(), mask=["integer", "float"]),
            dict(
                algo_name="PYMOO_NSGA2",
                crossover=dict(
                    integer=dict(custom=get_mutation("int_pm")), float="real_sbx"
                ),
            ),
            pytest.raises(Exception, match="must be an instance of"),
        ),
        (
            Rosenbrock(),
            dict(
                algo_name="PYMOO_NSGA3",
                ref_dirs_name="layer-energy",
                partitions=array([1, 5]),
            ),
            pytest.raises(
                ValueError,
                match="For a single-objective problem, "
                "the partitions array must be of size 1",
            ),
        ),
        (
            Viennet(),
            dict(algo_name="PYMOO_NSGA2", mutation=dict(custom=DummyMutation())),
            pytest.raises(
                Exception, match=r"\D+ must be an instance of \D+ or inherit from it!"
            ),
        ),
    ],
)
def test_execution_exceptions(opt_factory, problem, options, expectation):
    """Test different exceptions raised during the optimization execution.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        problem: The problem to be optimized.
        options: The options for the optimization execution.
        expectation: The expected exception to be raised.
    """
    with expectation:
        opt_factory.execute(problem, **options)


@pytest.mark.parametrize(
    "operators",
    [
        {},
        dict(sampling="real_lhs", crossover=("real_k_point", [2]), mutation="real_pm"),
    ],
)
def test_check_operator_suitability(opt_factory, pow2_ineq_int, operators, caplog):
    """Test the suitability of the operators for the given problem.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        pow2_ineq_int: Fixture returning the problem to be optimized.
        operators: A dictionary with the evolutionary operators.
        caplog: Fixture to access and control log capturing.
    """
    options = dict(max_iter=100, **tolerances, **operators, **integer_options)
    opt_factory.execute(pow2_ineq_int, algo_name="PYMOO_NSGA2", **options)

    assert "operator may not be suitable for integer" in caplog.text


def test_empty_database(opt_factory):
    """Test an empty multi-objective optimization result.

    Args:
        opt_factory: Fixture returning an optimizer factory.
    """
    opt = opt_factory.create("PYMOO_NSGA2")

    # Assign an optimization problem.
    opt.problem = Viennet()

    res = opt.get_optimum_from_database()
    s_res = str(res)
    r_res = str(repr(res))

    assert "The result is unfeasible." in s_res
    assert "Number of calls to the objective function by the optimizer: 0" in s_res
    assert "Pareto available: False" in r_res

    # Check if dictionary does not contain any pareto related attribute.
    assert "pareto" not in str(res.to_dict().keys())


def test_hypervolume_check_particularities(opt_factory, mo_knapsack, caplog):
    """Test the hypervolume stop criterion with an unfeasible problem.

    Args:
        opt_factory: Fixture returning an optimizer factory.
        mo_knapsack: Fixture returning the problem to be optimized.
        caplog: Fixture to access and control log capturing.
    """
    caplog.set_level(logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Set the knapsack problem to be unfeasible.
    mo_knapsack.capacity_items = -1

    options = dict(max_gen=6, pop_size=2, **integer_operators, **integer_options)
    opt_factory.execute(mo_knapsack, algo_name="PYMOO_NSGA2", **options)

    assert "Current hypervolume set to 0!" in caplog.text
    assert "Hypervolume stopping criterion is ignored!" in caplog.text
