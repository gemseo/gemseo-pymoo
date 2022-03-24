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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
from __future__ import annotations

import numpy as np
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.api import configure_logger
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo_pymoo.algos.opt.lib_pymoo import get_gemseo_opt_problem

configure_logger()


def test_rnsga3_rosen():
    problem = Rosenbrock()
    options = {
        "mu": 0.001,
        "max_iter": 2000,
        "verbose": True,
        "pop_size": 20,
        "ref_points": np.array([[1.0], [0.0]]),
        "ftol_rel": 0.0,
        "ftol_abs": 0.0,
        "xtol_rel": 0.0,
        "xtol_abs": 0.0,
    }
    res = OptimizersFactory().execute(problem, algo_name="RNSGA3", **options)

    assert abs(res.f_opt) < 5e-3


def test_rnsga3():
    problem = Power2()
    problem.constraints = problem.get_ineq_constraints()
    options = {
        "mu": 0.1,
        "max_iter": 10000,
        "ref_points": np.array([[1.0], [10.0]]),
        "pop_size": 20,
        "ftol_rel": 0.0,
        "ftol_abs": 0.0,
        "xtol_rel": 0.0,
        "xtol_abs": 0.0,
        "ineq_tolerance": 1e-3,
    }
    res = OptimizersFactory().execute(problem, algo_name="RNSGA3", **options)

    assert abs(1.26 - res.f_opt) < 1e-3


def test_get_gemseo_opt_problem():
    problem = get_gemseo_opt_problem("rosenbrock")
    res = OptimizersFactory().execute(problem, algo_name="L-BFGS-B")
    assert abs(res.f_opt) < 1e-5
