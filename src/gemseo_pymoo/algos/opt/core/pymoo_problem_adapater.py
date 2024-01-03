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
"""An adapter for pymoo :class:`~pymoo.core.problem.Problem`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Union

import h5py
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.stop_criteria import TerminationCriterion
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from numpy import allclose
from numpy import atleast_1d
from numpy import average
from numpy import concatenate as np_concat
from numpy import dtype as np_dtype
from numpy import hstack
from numpy import inf as np_inf
from numpy import max as np_max
from numpy import ndarray
from numpy import ones
from numpy import vstack
from numpy import zeros
from pymoo.core.problem import Problem
from pymoo.indicators.hv import Hypervolume
from pymoo.problems import get_problem

from gemseo_pymoo.algos.opt_result_mo import MultiObjectiveOptimizationResult
from gemseo_pymoo.algos.opt_result_mo import Pareto
from gemseo_pymoo.algos.stop_criteria import HyperVolumeToleranceReached
from gemseo_pymoo.algos.stop_criteria import MaxGenerationsReached

if TYPE_CHECKING:
    from gemseo.algos.opt.optimization_library import OptimizationLibrary

LOGGER = logging.getLogger(__name__)
OPTLibraryOutputType = tuple[dict[str, Union[float, ndarray]], dict[str, ndarray]]


def import_hdf(
    file_path: str,
    x_tolerance: float = 0.0,
) -> OptimizationProblem:
    """Import an optimization history from an HDF file.

    It uses :meth:`gemseo.algos.opt_problem.OptimizationProblem.from_hdf` to import
    the optimization history from the HDF file. Afterwards, it replaces the
    :class:`gemseo.algos.opt_result.OptimizationResult` object in
    :attr:`gemseo.algos.opt_problem.OptimizationProblem.solution` by a
    :class:`gemseo_pymoo.algos.opt_result_mo.MultiObjectiveOptimizationResult` object.

    Args:
        file_path: The file containing the optimization history.
        x_tolerance: The tolerance on the design variables when reading the file.

    Returns:
        The read optimization problem.
    """
    opt_pb = OptimizationProblem.from_hdf(file_path, x_tolerance)

    with h5py.File(file_path, "r") as h5file:
        # Update solution.
        if opt_pb.SOLUTION_GROUP in h5file:
            pareto = Pareto(opt_pb) if opt_pb.solution.is_feasible else None
            opt_pb.solution = MultiObjectiveOptimizationResult(
                **opt_pb.solution.__dict__, pareto=pareto
            )

    return opt_pb


def get_gemseo_opt_problem(
    pymoo_pb: str | Problem,
    **pymoo_pb_options: Any,
) -> OptimizationProblem:
    """Create a GEMSEO problem from a pymoo :class:`~pymoo.core.problem.Problem`.

    If the pymoo problem's name is provided, it must be a valid name among the available
    ones (see `pymoo problems <https://pymoo.org/problems/test_problems.html>`_).

    Args:
        pymoo_pb: The pymoo problem to be converted.
        **pymoo_pb_options: The additional arguments to be passed to the
            pymoo's problem ``getter`` ``get_problem``.

    Returns:
        An instance of a GEMSEO :class:`~gemseo.algos.opt_problem.OptimizationProblem`.

    Raises:
        TypeError: If ``pymoo_pb`` is not a valid string nor an instance of
            :class:`~pymoo.core.problem.Problem`.
    """
    if isinstance(pymoo_pb, str):
        pymoo_pb = get_problem(pymoo_pb, **pymoo_pb_options)

    if not isinstance(pymoo_pb, Problem):
        raise TypeError(f"Problem must be an instance of {Problem}!")

    design_space = DesignSpace()
    design_space.add_variable(
        "x",
        l_b=pymoo_pb.xl,
        u_b=pymoo_pb.xu,
        var_type=pymoo_pb_options.pop("mask", "float"),
        value=0.5 * (pymoo_pb.xl + pymoo_pb.xu),
        size=pymoo_pb.n_var,
    )

    gemseo_pb = OptimizationProblem(
        design_space,
        differentiation_method=OptimizationProblem.ApproximationMode.FINITE_DIFFERENCES,
    )
    gemseo_pb.objective = MDOFunction(
        lambda x: pymoo_pb.evaluate(x, return_as_dictionary=True)["F"], "F"
    )
    if pymoo_pb.n_constr > 0:
        ineq = MDOFunction(
            lambda x: pymoo_pb.evaluate(x, return_as_dictionary=True)["G"], "G"
        )
        gemseo_pb.add_constraint(ineq, cstr_type="ineq")
    return gemseo_pb


class PymooProblem(Problem):
    """Interface between GEMSEO and pymoo optimization problems.

    It supports multiprocessing.
    """

    opt_problem: OptimizationProblem
    """The GEMSEO optimization problem."""

    normalize_ds: bool
    """Whether the design space is normalized."""

    max_gen: int
    """The maximum number of generations allowed."""

    _driver: OptimizationLibrary
    """The optimization library currently handling the problem."""

    _n_gen: int
    """A counter to track the number of generations evaluated."""

    _hv_tol_rel: float
    """The relative tolerance to use in the hypervolume convergence check."""

    _hv_tol_abs: float
    """The absolute tolerance to use in the hypervolume convergence check."""

    _stop_crit_n_hv: float
    """The number of generations to account for in the hypervolume convergence check."""

    _hv_obj_hist_feasible: list[ndarray]
    """The objectives' value of the feasible points up to the current generation."""

    _hv_history: list[float]
    """The hypervolume indicator for each generation."""

    _hv_ref_point: ndarray
    """The reference point used for computing the hypervolume indicator."""

    _parallel: CallableParallelExecution | None
    """The object handling the parallel execution."""

    _ineq_constraints: list[MDOFunction]
    """The problem's inequality constraints."""

    def __init__(
        self,
        opt_problem: OptimizationProblem,
        normalize_ds: bool,
        driver: OptimizationLibrary,
        **options: Any,
    ) -> None:
        """Initialize a pymoo :class:`~pymoo.core.problem.Problem` from a GEMSEO one.

        It also sets up a parallel object
        :class:`~gemseo.core.parallel_execution.ParallelExecution` for
        multiprocessing purposes.

        Args:
            opt_problem: The GEMSEO problem to convert to a pymoo problem.
            normalize_ds: Whether to normalize the design variables.
            driver: The optimization library used to handle the problem.
            **options: The other algorithm options.
        """
        self.opt_problem = opt_problem
        self.normalize_ds = normalize_ds
        self._driver = driver

        # Track the number of generations.
        self._n_gen = 0
        self.max_gen = options.pop("max_gen")

        # Track the hypervolume indicator.
        self._hv_tol_rel = options.pop("hv_tol_rel", 1e-9)
        self._hv_tol_abs = options.pop("hv_tol_abs", 1e-9)
        self._stop_crit_n_hv = options.pop("stop_crit_n_hv", 5)
        self._hv_obj_hist_feasible = []
        self._hv_history = []
        self._hv_ref_point = -np_inf * ones(opt_problem.objective.dim, dtype=float)

        # Set up for parallel execution.
        n_processes = options.pop("n_processes")
        if n_processes > 1:
            LOGGER.info(
                "Running Optimization in parallel on n_processes = %d", n_processes
            )
            self._parallel = CallableParallelExecution(
                [self._worker],
                n_processes=n_processes,
                exceptions_to_re_raise=(TerminationCriterion,),
            )
        else:
            self._parallel = None

        # Design variables.
        design_space = opt_problem.design_space
        n_var = design_space.dimension
        if normalize_ds:
            lower_bounds = zeros(n_var)
            upper_bounds = ones(n_var)
        else:
            lower_bounds = design_space.get_lower_bounds()
            upper_bounds = design_space.get_upper_bounds()

        # Constraints.
        self._ineq_constraints = self.opt_problem.get_ineq_constraints()

        super().__init__(
            n_var=n_var,
            n_obj=opt_problem.objective.dim,
            n_constr=sum(constr.dim for constr in self._ineq_constraints),
            xl=lower_bounds,
            xu=upper_bounds,
            type_var=np_concat([
                design_space.get_type(var) for var in design_space.variable_names
            ]),
            **options,
        )

    def _evaluate(
        self,
        design_variables: ndarray,
        output_data: dict[str, ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Evaluate the objectives and constraints for the given design_variables.

        Args:
            design_variables: The design variables vector.
            output_data: The output data with objectives and constraints evaluations.
            *args: The optional arguments for the evaluation
                of objectives and constraints.
            **kwargs: The optional keyword arguments for the evaluation
                of objectives and constraints.
        """
        # When treating mixed variables, cast x array type from 'object' to 'float'.
        if design_variables.dtype == np_dtype(object):
            design_variables = design_variables.astype(float)

        if self._parallel is not None:
            obj, cstrs = self._evaluate_parallel(design_variables)
        else:
            # Constraints and objectives have to be inside the same 'for loop',
            # and the constraints must be calculated first.
            # This will ensure they are all calculated for each iteration and
            # before any termination criteria.
            obj, cstrs = [], []
            for x_i in design_variables:
                if self.n_constr:
                    cstrs.append(
                        hstack([atleast_1d(g(x_i)) for g in self._ineq_constraints])
                    )
                obj.append(self.opt_problem.objective(x_i))

        output_data["F"] = vstack(obj)
        if self.n_constr:
            output_data["G"] = vstack(cstrs)

        # Update the generation number.
        self._n_gen += 1

        # Check for termination criteria.
        self._new_generation_callback(obj)

    def _evaluate_parallel(
        self, design_variables: ndarray
    ) -> tuple[list[ndarray], list[ndarray]]:
        """Evaluate the objectives and constraints using multiple processes.

        Before evaluating the functions, all the input points are stored in the
        database. This allows to respect their order as it cannot be ensured during the
        parallel computations.

        Args:
            design_variables: The design variables vector.

        Returns:
            The objectives and the constraints evaluations.
        """
        sample_to_design = self.opt_problem.design_space.untransform_vect
        round_vect = self.opt_problem.design_space.round_vect
        database = self.opt_problem.database

        # Initialize the order as it is not necessarily guaranteed
        # when using parallel execution.
        for x_i in design_variables:
            if self.normalize_ds:
                x_u = sample_to_design(x_i)
                x_r = round_vect(x_u)
                database.store(x_r, {})
            else:
                database.store(x_i, {})

        # Define a callback function to store the samples on the fly
        # during the parallel execution.
        def store_callback(index: int, outputs: OPTLibraryOutputType) -> None:
            """Store the outputs in the database.

            The jacobian is ignored because we are dealing
            with gradient-free algorithms.

            Args:
                index: The sample index.
                outputs: The outputs of the parallel execution.
            """
            outs, _ = outputs
            if self.normalize_ds:
                x_u_ = sample_to_design(design_variables[index])
                x_r_ = round_vect(x_u_)
            else:
                x_r_ = design_variables[index]
            database.store(x_r_, outs)

        # The list of inputs of the tasks is the list of samples.
        self._parallel.execute(design_variables, exec_callback=store_callback)

        # We added empty entries by default to keep order in the database
        # but when the calculation point is failed, this is not consistent
        # with the serial exec, so we clean the database.
        database.remove_empty_entries()

        # Retrieve objectives and constraints from database.
        obj, cstrs = [], []
        for x_i in design_variables:
            # Unormalize vector to access the right key in the database.
            if self.normalize_ds:
                x_u = sample_to_design(x_i)
                x_i = round_vect(x_u)

            # Dictionary with all functions evaluation for the current key x_i.
            funcs = database.get(x_i)

            if self.n_constr:
                c_values = [
                    atleast_1d(funcs.get(g.name)) for g in self._ineq_constraints
                ]
                cstrs.append(hstack(c_values))
            obj.append(funcs.get(self.opt_problem.objective.name))

        return obj, cstrs

    def _worker(self, sample: ndarray) -> OPTLibraryOutputType:
        """Wrap the evaluation of the functions for parallel execution.

        Args:
            sample: The values for the evaluation of the functions.

        Returns:
            The computed values.
        """
        # No need to check subprocess name,
        # since it is set by the ParallelExecution class and must not change.
        self._driver.deactivate_progress_bar()
        self.opt_problem.database.clear_listeners()
        return self.opt_problem.evaluate_functions(
            x_vect=sample, normalize=self.normalize_ds
        )

    def _new_generation_callback(self, obj: list[ndarray]) -> None:
        """Callback called at each new generation evaluated.

        Args:
            obj: The objective value for all the individuals in the generation.

        Raises:
            MaxGenerationsReached: If the maximum number of generations is reached.
            HyperVolumeToleranceReached: If the hypervolume indicator has converged.
        """
        # Termination criterion on the number of generations.
        # It is important to avoid being stuck when dealing with discrete variables,
        # because max_iter may take too long to be reached given the modus operandi
        # of GAs and the way GEMSEO counts the number of iterations.
        if self._n_gen == self.max_gen:
            raise MaxGenerationsReached

        obj_name = self.opt_problem.objective.name

        # Filter only the feasible points because this is not done by pymoo.
        # Nevertheless, pymoo will check and select the non-dominated points
        # thanks to the attribute 'nds'. In this way, we do not have to calculate
        # the pareto front at every generation.
        _, f_hist_feasible = self.opt_problem.get_feasible_points()
        if f_hist_feasible:
            obj_hist_feasible = vstack([f_val[obj_name] for f_val in f_hist_feasible])
        else:
            LOGGER.debug(
                "Generation %d does not yield any feasible solution. "
                "Current hypervolume set to 0!",
                self._n_gen,
            )
            # Give an infinity value to the objective(s)
            # will lead to a hypervolume of 0.
            obj_hist_feasible = np_inf * ones(self.n_obj)
        self._hv_obj_hist_feasible.append(obj_hist_feasible)

        # Get the reference point (nadir point) of the alltime objective history.
        new_hv_ref_point = np_max(vstack([self._hv_ref_point, *obj]), axis=0)

        # At the first generation, the reference point is not yet well-defined.
        if self._n_gen == 1:
            self._hv_ref_point = new_hv_ref_point

        # If the reference point has changed,
        # we must recalculate the hypervolume history for all the past generations.
        if not all(new_hv_ref_point == self._hv_ref_point):
            LOGGER.debug(
                "The hypervolume reference point changed from %s to %s",
                self._hv_ref_point,
                new_hv_ref_point,
            )
            self._hv_ref_point = new_hv_ref_point
            hyper_volume = Hypervolume(ref_point=new_hv_ref_point, nds=True)

            for gen_i in range(self._n_gen - 1):
                obj_history = self._hv_obj_hist_feasible[gen_i]
                self._hv_history[gen_i] = hyper_volume.do(obj_history)

            LOGGER.debug("The hypervolume history has been updated.")

        # Compute the hypervolume indicator for the current generation.
        hyper_volume = Hypervolume(ref_point=self._hv_ref_point, nds=True)
        self._hv_history.append(hyper_volume.do(obj_hist_feasible))
        LOGGER.debug(
            "Hypervolume at generation %d: %g", self._n_gen, self._hv_history[-1]
        )

        # Run for at least 'stop_crit_n_hv' generations.
        if self._n_gen < self._stop_crit_n_hv:
            return

        # Get last 'stop_crit_n_hv' hypervolume values and average it.
        n_last_hv = self._hv_history[-self._stop_crit_n_hv :]
        hv_average = average(n_last_hv)

        # Wait until at least one generation have feasible individuals,
        # and therefore a non-zero hypervolume is available.
        if hv_average == 0:
            LOGGER.debug(
                "Last %d generations did not yield any feasible solution. "
                "Hypervolume stopping criterion is ignored!",
                self._stop_crit_n_hv,
            )
            return

        # Termination criterion based on the convergence of the pareto front.
        if allclose(
            n_last_hv, hv_average, atol=self._hv_tol_abs, rtol=self._hv_tol_rel
        ):
            raise HyperVolumeToleranceReached
