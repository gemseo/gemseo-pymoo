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
"""Tests for the pymoo post-processing."""
from __future__ import annotations

from pathlib import Path

import pytest
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.power_2 import Power2
from gemseo.utils.testing import image_comparison
from gemseo_pymoo.post.scatter_pareto import ScatterPareto
from gemseo_pymoo.problems.analytical.chankong_haimes import ChankongHaimes
from gemseo_pymoo.problems.analytical.viennet import Viennet
from numpy import array


@pytest.fixture
def problem_1obj() -> OptimizationProblem:
    """Create an optimization problem with 1 objective ready to be post-processed.

    Returns:
        A :class:`.Power2` instance.
    """
    power2 = Power2()
    power2.constraints = power2.get_ineq_constraints()
    OptimizersFactory().execute(power2, algo_name="PYMOO_NSGA2", max_iter=700)
    return power2


@pytest.fixture
def problem_2obj() -> OptimizationProblem:
    """Create an optimization problem with 2 objectives ready to be post-processed.

    Returns:
        A :class:`.ChankongHaimes` instance.
    """
    problem = ChankongHaimes()
    OptimizersFactory().execute(problem, algo_name="PYMOO_NSGA2", max_iter=700)
    return problem


@pytest.fixture
def problem_3obj() -> OptimizationProblem:
    """Create an optimization problem with 3 objectives ready to be post-processed.

    Returns:
        A :class:`.Viennet` instance.
    """
    problem = Viennet()
    OptimizersFactory().execute(problem, algo_name="PYMOO_NSGA2", max_iter=700)
    return problem


@pytest.fixture
def post_factory() -> PostFactory:
    """Create a :class:`gemseo.post.post_factory.PostFactory` instance.

    Returns:
        A post-processing factory instance.
    """
    return PostFactory()


def test_saving(tmp_wd, post_factory, problem_2obj, pyplot_close_all):
    """Test the figure saving.

    Args:
        tmp_wd: Fixture to move into a temporary working directory.
        post_factory: Fixture returning a post-processing factory.
        problem_2obj: Fixture returning the optimization problem to be post-processed.
        pyplot_close_all: Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    options = dict(scalar_name="asf", weights=[0.3, 0.7], plot_arrow=True)
    post = post_factory.execute(
        problem_2obj, "Compromise", save=True, file_path="compromise1", **options
    )
    assert len(post.output_files) == 1
    for file in post.output_files:
        assert Path(file).exists()


@pytest.mark.parametrize(
    "diagram_name, s_func, opts, baseline_images",
    [
        ("Petal", "weighted-sum", {}, ["petal_viennet_weighted_sum"]),
        ("Petal", "tchebi", {}, ["petal_viennet_tchebi"]),
        ("Radar", "pbi", {}, ["radar_viennet_pbi"]),
        ("Radar", "asf", {}, ["radar_viennet_asf"]),
        ("ScatterPareto", "", {"plot_arrow": True}, ["scatter_pareto_viennet"]),
        ("Compromise", "aasf", {"plot_arrow": True}, ["compromise_viennet_aasf"]),
        (
            "Compromise",
            "perp_dist",
            {"plot_arrow": False},
            ["compromise_viennet_perp_dist"],
        ),
        ("HighTradeOff", "", {"plot_extra": False}, ["high_tradeoff_viennet_no_extra"]),
    ],
)
@image_comparison(None, extensions=["png"], style="default")
def test_post(
    post_factory,
    problem_3obj,
    diagram_name,
    s_func,
    opts,
    baseline_images,
    pyplot_close_all,
):
    """Test images created by the post-processes.

     The new images are compared with existing references.

    Args:
        post_factory: Fixture returning a post-processing factory.
        problem_3obj: Fixture returning the optimization problem to be post-processed.
        diagram_name: The name of the diagram.
        s_func: The name of the scalarization function.
        opts: The post-processing options.
        baseline_images: The reference images to be compared.
        pyplot_close_all: Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    options = dict(file_extension="png", save=False, **opts)
    if diagram_name not in ["HighTradeOff", "ScatterPareto"]:
        options.update(
            **dict(
                scalar_name=s_func,
                weights=[[0.3, 0.5, 0.7], [0.5, 0.3, 0.7], [0.5, 0.7, 0.3]],
                beta=5,
            )
        )

    # Check "weights = None" option.
    if diagram_name == "Compromise" and s_func == "perp_dist":
        options.pop("weights")

    post = post_factory.execute(problem_3obj, diagram_name, **options)

    # Cover Arrow3D and Annotation3D draw methods.
    if diagram_name == "Compromise" and opts["plot_arrow"]:
        fig_name = f"compromise_{s_func}_1"
        post.figures[fig_name].draw(post.figures[fig_name].canvas.get_renderer())

    post.figures


@pytest.mark.parametrize(
    "single_objective, options, expectation",
    [
        (
            True,
            dict(points=array([[0, 1], [2, 3], [4, 5]]), point_labels=["a", "b"]),
            pytest.raises(
                ValueError,
                match="This post-processing is only suitable for optimization "
                "problems with 2 or 3 objective functions!",
            ),
        ),
        (
            False,
            dict(points=array([[0, 1], [2, 3], [4, 5]]), point_labels=["a", "b"]),
            pytest.raises(
                ValueError,
                match="You must provide either a single label for all points "
                "or one label for each one!",
            ),
        ),
    ],
)
def test_exceptions_scatter(
    problem_1obj, problem_2obj, single_objective, options, expectation
):
    """Test different exceptions raised during the Scatter post-processing execution.

    Args:
        problem_1obj: Fixture returning the single-objective
            optimization problem to be post-processed.
        problem_2obj: Fixture returning the multi-objective
            optimization problem to be post-processed.
        single_objective: Whether to use the single-objective optimization problem.
        options: The post-processing options.
        expectation: The expected exception to be raised.
    """
    if single_objective:
        problem = problem_1obj
    else:
        problem = problem_2obj

    post = ScatterPareto(problem)
    with expectation:
        post.execute(save=False, **options)


@pytest.mark.parametrize(
    "options, expectation",
    [
        (
            dict(scalar_name="unknown", weights=[1]),
            pytest.raises(
                ValueError,
                match="The scalarization function name must be one of the following",
            ),
        ),
        (
            dict(scalar_name="asf", weights=[1, 2]),
            pytest.raises(
                ValueError,
                match="You must provide exactly one weight for each objective function",
            ),
        ),
    ],
)
def test_exceptions_compromise(post_factory, problem_1obj, options, expectation):
    """Test different exceptions raised during the Compromise post-processing execution.

    Args:
        post_factory: Fixture returning a post-processing factory.
        problem_1obj: Fixture returning the optimization problem to be post-processed.
        options: The post-processing options.
        expectation: The expected exception to be raised.
    """
    with expectation:
        post_factory.execute(problem_1obj, "Compromise", save=False, **options)


@pytest.mark.parametrize(
    "diagram, options, expectation",
    [
        (
            "Petal",
            dict(scalar_name="unknown", weights=[1, 2]),
            pytest.raises(
                ValueError,
                match="The scalarization function name must be one of ",
            ),
        ),
        (
            "Petal",
            dict(scalar_name="asf", weights=[1, 2, 3]),
            pytest.raises(
                ValueError,
                match="provide exactly one weight for each objective",
            ),
        ),
        (
            "Radar",
            dict(scalar_name="asf", weights=[1, 2]),
            pytest.raises(
                ValueError,
                match="The Radar post-processing is only suitable for optimization "
                "problems with at least 3 objective functions!",
            ),
        ),
    ],
)
def test_exceptions_diagrams(post_factory, problem_2obj, diagram, options, expectation):
    """Test different exceptions raised during the Diagrams post-processing execution.

    Args:
        post_factory: Fixture returning a post-processing factory.
        problem_2obj: Fixture returning the optimization problem to be post-processed.
        diagram: The type of diagram to be created.
        options: The post-processing options.
        expectation: The expected exception to be raised.
    """
    with expectation:
        post_factory.execute(problem_2obj, diagram, save=False, **options)
