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
#        :author: Francois Gallard
"""pymoo optimization library wrapper."""

import logging
from typing import Any
from typing import Dict
from typing import Union

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from numpy import hstack
from numpy import ndarray
from numpy import ones
from numpy import zeros
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.core.population import Population
from pymoo.core.problem import Problem as PyMooOptProblem
from pymoo.core.sampling import Sampling
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

LOGGER = logging.getLogger(__name__)


class GEMSEOPyMooProblem(PyMooOptProblem):
    def __init__(self, opt_problem, n_obj, normalize_ds):
        design_space = opt_problem.design_space
        n_var = opt_problem.design_space.dimension
        self.opt_problem = opt_problem
        if normalize_ds:
            x_l = zeros(n_var)
            x_u = ones(n_var)
        else:
            x_l = design_space.get_lower_bounds()
            x_u = design_space.get_upper_bounds()
        super().__init__(n_var=n_var, n_obj=n_obj, xl=x_l, xu=x_u)

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] = []
        for x_i in (x[k] for k in range(len(x))):
            out["F"].append(self.opt_problem.objective(x_i))
        out["F"] = hstack(out["F"])

        if not self.opt_problem.has_constraints():
            return

        out["G"] = []
        for x_i in (x[k] for k in range(len(x))):
            cstrs_vals = [
                constr(x_i) for constr in self.opt_problem.get_ineq_constraints()
            ]
            out["G"].append(cstrs_vals)

        out["G"] = hstack(out["G"]).T


def get_gemseo_opt_problem(pymoo_problem_name, **kwargs):
    pymoo_pb = get_problem(pymoo_problem_name, **kwargs)
    design_space = DesignSpace()
    design_space.add_variable(
        "x",
        l_b=pymoo_pb.xl,
        u_b=pymoo_pb.xu,
        value=0.5 * (pymoo_pb.xl + pymoo_pb.xu),
        size=pymoo_pb.n_var,
    )

    gemseo_pb = OptimizationProblem(
        design_space, differentiation_method="finite_differences"
    )
    obj = MDOFunction(
        lambda x: pymoo_pb.evaluate(x, return_as_dictionary=True)["F"], "F"
    )
    gemseo_pb.objective = obj
    if pymoo_pb.n_constr > 0:
        ineq = MDOFunction(
            lambda x: pymoo_pb.evaluate(x, return_as_dictionary=True)["G"], "G"
        )
        gemseo_pb.add_constraint(ineq, cstr_type="ineq")
    return gemseo_pb


class PyMoo(OptimizationLibrary):
    """Scipy optimization library interface.

    See OptimizationLibrary.
    """

    OPTIONS_MAP = {}

    def __init__(self):
        """Constructor.

        Generate the library dict, contains the list
        of algorithms with their characteristics:

        - does it require gradient
        - does it handle equality constraints
        - does it handle inequality constraints
        """
        super().__init__()
        doc = "https://www.pymoo.org/algorithms"

        # See https://www.pymoo.org/misc/constraints.html?highlight=out%20h
        # for eq constraints

        self.lib_dict = {
            "UNSGA3": {
                self.INTERNAL_NAME: "UNSGA3",
                self.REQUIRE_GRAD: False,
                self.POSITIVE_CONSTRAINTS: False,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: True,
                self.DESCRIPTION: "",
                self.WEBSITE: doc + "/moo/unsga3.html#nb-unsga3",
            },
            "RNSGA3": {
                self.INTERNAL_NAME: "RNSGA3",
                self.REQUIRE_GRAD: False,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: True,
                self.POSITIVE_CONSTRAINTS: False,
                self.DESCRIPTION: "",
                self.WEBSITE: doc + "/moo/rnsga3.html#nb-rnsga3",
            },
        }

    def _get_options(
        self,
        max_iter=999,  # type: int
        ftol_rel=1e-9,  # type: float
        ftol_abs=1e-9,  # type: float
        xtol_rel=1e-9,  # type: float
        xtol_abs=1e-9,  # type: float
        normalize_design_space=True,  # type: int
        eq_tolerance=1e-2,  # type: float
        ineq_tolerance=1e-4,  # type: float
        n_obj=1,  #
        ref_dirs=None,  # type: ndarray
        pop_size=None,  # type: int
        sampling=None,  # type: Union[Sampling, ndarray, Population]
        selection=None,
        mutation=None,
        crossover=None,
        seed=1,
        pop_per_ref_point=1,
        mu=0.1,
        n_pareto_points=50,
        ref_points=None,
        verbose=False,
        **kwargs,  # type: Any
    ):  # type: (...) -> Dict[str, Any]
        r"""Set the options default values.

        To get the best and up to date information about algorithms options,
        go to scipy.optimize documentation:
        https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

        Args:
            max_iter: The maximum number of iterations, i.e. unique calls to f(x).
            ftol_rel: A stop criteria, the relative tolerance on the
               objective function.
               If abs(f(xk)-f(xk+1))/abs(f(xk))<= ftol_rel: stop.
            ftol_abs: A stop criteria, the absolute tolerance on the objective
               function. If abs(f(xk)-f(xk+1))<= ftol_rel: stop.
            xtol_rel: A stop criteria, the relative tolerance on the
               design variables. If norm(xk-xk+1)/norm(xk)<= xtol_rel: stop.
            xtol_abs: A stop criteria, absolute tolerance on the
               design variables.
               If norm(xk-xk+1)<= xtol_abs: stop.
            n_obj: The number of objectives.
            normalize_design_space: If True, scales variables to [0, 1].
            eq_tolerance: The equality tolerance.
            ineq_tolerance: The inequality tolerance.
            ref_dirs: The reference directions.
            pop_size: The population size.
            sampling: The sampling process that generates the initial population.
            selection: The mating selection operator.
            crossover: The crossver operator used to create offsprings.
            mutation: The mutation operator in the GA.
            n_offsprings: Number of offspring that are created through mating.
            **kwargs: The other algorithm options.
        """
        nds = normalize_design_space
        popts = self._process_options(
            max_iter=max_iter,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            normalize_design_space=nds,
            ineq_tolerance=ineq_tolerance,
            eq_tolerance=eq_tolerance,
            selection=selection,
            sampling=sampling,
            mutation=mutation,
            crossover=crossover,
            seed=seed,
            mu=mu,
            n_obj=n_obj,
            ref_points=ref_points,
            pop_per_ref_point=pop_per_ref_point,
            pop_size=pop_size,
            n_pareto_points=n_pareto_points,
            verbose=verbose,
            **kwargs,
        )
        return popts

    def _run(
        self, **options  # type: Any
    ):  # type: (...) -> OptimizationResult
        """Run the algorithm, to be overloaded by subclasses.

        Args:
            **options: The options for the algorithm.

        Returns:
            The optimization result.
        """
        # remove normalization from options for algo
        normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)
        # Get the normalized bounds:
        n_obj = options.pop("n_obj")
        pymoo_problem = GEMSEOPyMooProblem(
            self.problem, n_obj=n_obj, normalize_ds=normalize_ds
        )
        if n_obj > 1:
            ref_dirs = UniformReferenceDirectionFactory(
                n_obj, n_points=options.pop("n_pareto_points")
            ).do()
            pareto_front = pymoo_problem.pareto_front(ref_dirs)
        else:
            pareto_front = None
        if self.internal_algo_name == "RNSGA3":
            algorithm = RNSGA3(
                ref_points=options.pop("ref_points"),
                pop_per_ref_point=options.pop("pop_per_ref_point"),
                mu=options.pop("mu"),
            )

        res = minimize(
            pymoo_problem,
            algorithm=algorithm,
            termination=("n_gen", 10000000),
            pf=pareto_front,
            **options,
        )

        reference_directions = res.algorithm.survival.ref_dirs
        gems_res = self.get_optimum_from_database(res.message, res.success)
        gems_res.reference_directions = reference_directions
        gems_res.pareto_front = pareto_front
        return gems_res
