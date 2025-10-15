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
"""Pymoo optimization library wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from gemseo.algos.multiobjective_optimization_result import (
    MultiObjectiveOptimizationResult,
)
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.algos.optimization_problem import OptimizationProblem  # noqa: TC002
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.algos.stop_criteria import TerminationCriterion  # noqa: TC002
from numpy import inf
from numpy import prod as np_prod
from numpy import size as np_size
from numpy import unique as np_unique
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.operator import Operator  # noqa: TC002
from pymoo.core.sampling import Sampling
from pymoo.core.selection import Selection
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.reference_direction import (
    MultiLayerReferenceDirectionFactory,  # noqa: TC002
)
from pymoo.util.reference_direction import ReferenceDirectionFactory  # noqa: TC002

from gemseo_pymoo.algos.opt._base_pymoo_settings import BasePymooSettings
from gemseo_pymoo.algos.opt._settings.ga_settings import GASettings
from gemseo_pymoo.algos.opt._settings.nsga2_settings import NSGA2Settings
from gemseo_pymoo.algos.opt._settings.nsga3_settings import NSGA3Settings
from gemseo_pymoo.algos.opt._settings.rnsga3_settings import RNSGA3Settings
from gemseo_pymoo.algos.opt._settings.unsga3_settings import UNSGA3Settings
from gemseo_pymoo.algos.opt.core.pymoo_problem import PymooProblem
from gemseo_pymoo.algos.stop_criteria import DesignSpaceExploredException
from gemseo_pymoo.algos.stop_criteria import HyperVolumeToleranceReached
from gemseo_pymoo.algos.stop_criteria import MaxGenerationsReached

if TYPE_CHECKING:
    from numpy import ndarray

LOGGER = logging.getLogger(__name__)

EvolutionaryOperatorTypes = Crossover | Mutation | Sampling | Selection


@dataclass
class PymooAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of an optimization algorithm from the pymoo library."""

    library_name: str = "pymoo"

    handle_integer_variables: bool = True

    require_grad: bool = False

    handle_equality_constraints: bool = False

    handle_inequality_constraints: bool = True

    positive_constraints: bool = False

    for_linear_problems: bool = False

    Settings: type[BasePymooSettings] = BasePymooSettings
    """ "The option validation model for Gemseo Pymoo optimization library plugin."""


class PymooOpt(BaseOptimizationLibrary[BasePymooSettings]):
    """Pymoo optimization library interface.

    See :class:`gemseo.algos.opt.optimization_library.OptimizationLibrary`.
    """

    __DOC: Final[str] = "https://www.pymoo.org/algorithms/"

    N_PROCESSES: Final[str] = "n_processes"
    """The tag for the number of processes to use."""

    MAX_GEN: Final[str] = "max_gen"
    """The tag for the maximum number of generations allowed."""

    HV_TOL_REL: Final[str] = "hv_tol_rel"
    """The tag for the relative tolerance used in the hypervolume convergence check."""

    HV_TOL_ABS: Final[str] = "hv_tol_abs"
    """The tag for the absolute tolerance used in the hypervolume convergence check."""

    STOP_CRIT_N_HV: Final[str] = "stop_crit_n_hv"
    """The tag for the number of generations to account for in the hypervolume check."""

    __PYMOO_PREFIX: Final[str] = "PYMOO_"
    """The prefix added to the internal algorithm's name."""

    CROSSOVER_OPERATOR: Final[str] = "crossover"
    """The crossover operator's name."""

    MUTATION_OPERATOR: Final[str] = "mutation"
    """The mutation operator's name."""

    SAMPLING_OPERATOR: Final[str] = "sampling"
    """The sampling operator's name."""

    SELECTION_OPERATOR: Final[str] = "selection"
    """The selection operator's name."""

    EVOLUTIONARY_OPERATORS: Final[dict[str, Operator]] = {
        CROSSOVER_OPERATOR: Crossover,
        MUTATION_OPERATOR: Mutation,
        SAMPLING_OPERATOR: Sampling,
        SELECTION_OPERATOR: Selection,
    }
    """A dictionary with all evolutionary operators available as keys and their Pymoo
    classes as values."""

    PYMOO_GA: Final[str] = "PYMOO_GA"
    """The GEMSEO alias for the Genetic Algorithm."""

    PYMOO_NSGA2: Final[str] = "PYMOO_NSGA2"
    """The GEMSEO alias for the Non-dominated Sorting Genetic Algorithm II."""

    PYMOO_NSGA3: Final[str] = "PYMOO_NSGA3"
    """The GEMSEO alias for the Non-dominated Sorting Genetic Algorithm III."""

    PYMOO_UNSGA3: Final[str] = "PYMOO_UNSGA3"
    """The GEMSEO alias for the Unified NSGA-III."""

    PYMOO_RNSGA3: Final[str] = "PYMOO_RNSGA3"
    """The GEMSEO alias for the Reference Point Based NSGA-III."""

    __PYMOO_ = "PYMOO_"

    LIBRARY_NAME: Final[str] = "pymoo"
    """The library's name."""

    pymoo_n_gen: int = 10000000
    """The pymoo's termination criterion based on the number of generations."""

    _stop_crit_n_hv: int = 5
    """The number of generations to account for in the hypervolume convergence check."""

    _ds_size: int
    """The design space size."""

    _RESULT_CLASS: ClassVar[type[OptimizationResult]] = MultiObjectiveOptimizationResult
    """The class used to present the result of the optimization."""

    ALGORITHM_INFOS: ClassVar[dict[str, PymooAlgorithmDescription]] = {
        "PYMOO_GA": PymooAlgorithmDescription(
            algorithm_name="GA",
            description=("Genetic Algorithm (GA) implemented in the Pymoo library"),
            handle_equality_constraints=False,
            handle_inequality_constraints=True,
            internal_algorithm_name="GA",
            require_gradient=False,
            positive_constraints=True,
            handle_multiobjective=False,
            website=f"{__DOC}soo/ga.html",
            Settings=GASettings,
        ),
        "PYMOO_NSGA2": PymooAlgorithmDescription(
            algorithm_name="NSGA2",
            description=(
                "Non-Dominated Sorting Genetic Algorithm II (NSGA2) "
                "implemented in the Pymoo library"
            ),
            handle_equality_constraints=False,
            handle_inequality_constraints=True,
            internal_algorithm_name="NSGA2",
            require_gradient=False,
            positive_constraints=True,
            handle_multiobjective=True,
            website=f"{__DOC}moo/nsga2.html",
            Settings=NSGA2Settings,
        ),
        "PYMOO_NSGA3": PymooAlgorithmDescription(
            algorithm_name="NSGA3",
            description=(
                "Non-Dominated Sorting Genetic Algorithm III (NSGA3) "
                "implemented in the Pymoo library"
            ),
            handle_equality_constraints=False,
            handle_inequality_constraints=True,
            internal_algorithm_name="NSGA3",
            require_gradient=False,
            positive_constraints=True,
            handle_multiobjective=True,
            website=f"{__DOC}moo/nsga3.html",
            Settings=NSGA3Settings,
        ),
        "PYMOO_UNSGA3": PymooAlgorithmDescription(
            algorithm_name="UNSGA3",
            description=("Unified NSGA III implemented in the Pymoo library"),
            handle_equality_constraints=False,
            handle_inequality_constraints=True,
            internal_algorithm_name="UNSGA3",
            require_gradient=False,
            positive_constraints=True,
            handle_multiobjective=True,
            website=f"{__DOC}moo/unsga3.html",
            Settings=UNSGA3Settings,
        ),
        "PYMOO_RNSGA3": PymooAlgorithmDescription(
            algorithm_name="RNSGA3",
            description=(
                "Reference Point Based NSGA III implemented in the Pymoo library"
            ),
            handle_equality_constraints=False,
            handle_inequality_constraints=True,
            internal_algorithm_name="RNSGA3",
            require_gradient=False,
            positive_constraints=True,
            handle_multiobjective=True,
            website=f"{__DOC}moo/rnsga3.html",
            Settings=RNSGA3Settings,
        ),
    }

    def __init__(self, algo_name: str) -> None:
        """Constructor.

        Generate the library dict, which contains the list
        of algorithms with their characteristics:

        - does it require gradient
        - does it handle equality constraints
        - does it handle inequality constraints
        """
        super().__init__(algo_name)

        # The design space size (useful for finite discrete problems).
        self._ds_size = inf

    def _check_mo_handling(
        self,
        opt_problem: OptimizationProblem,
    ) -> None:
        """Check if the algorithm is capable of handling the optimization problem.

        All algorithms implemented are capable of handling  multi-objective problems,
        except the :class:`~pymoo.algorithms.soo.nonconvex.ga.GA` one.

        Args:
            opt_problem: The problem to be solved.

        Raises:
            ValueError: If the algo cannot handle the problem to be solved.
        """
        if (
            opt_problem.objective.dim > 1
            and not self.ALGORITHM_INFOS[self._algo_name].handle_multiobjective
        ):
            msg = (
                f"Requested optimization algorithm {self._algo_name} can not handle "
                "multiple objectives."
            )
            raise ValueError(msg)

    @staticmethod
    def _check_operator_suitable(
        lower_bounds: ndarray,
        upper_bounds: ndarray,
        operator_instance: EvolutionaryOperatorTypes,
        operator_class: Operator,
    ) -> None:
        """Check operator suitability according to design variables type and bounds.

        Args:
            lower_bounds: The design variables' lower bounds.
            upper_bounds: The design variables' upper bounds.
            operator_instance: The pymoo operator instance.
            operator_class: The specific class of operator that the
                ``operator_instance`` that is being checked should be an instance of.

        Raises:
            TypeError: If the ``operator_instance`` is not an instance of the required
                ``operator_class``.
            ValueError: If ``operator_instance`` refers to the mutation operator
                :class:`~pymoo.operators.mutation.pm.PolynomialMutation` and at least
                one design variable has equal lower and upper bounds.
        """
        if not isinstance(operator_instance, operator_class):
            msg = (
                f"{operator_instance.__class__.__name__} must be an instance of "
                f"{operator_class.__class__.__name__} or inherit from it."
            )
            raise TypeError(msg)

        # Anticipate 'division by zero' errors when using PolynomialMutation,
        # which occurs when we have equal lower and upper bounds.
        if isinstance(operator_instance, PolynomialMutation) and any(
            upper_bounds == lower_bounds
        ):
            msg = (
                "PolynomialMutation cannot handle equal lower and upper bounds.\n"
                "Consider setting those design variables as constants of your problem."
            )
            raise ValueError(msg)

    def _get_ref_dirs(
        self,
        ref_dirs_name: str,
        **ref_dirs_settings: dict[str, str],
    ) -> ReferenceDirectionFactory | MultiLayerReferenceDirectionFactory:
        r"""Return the reference directions.

        Get the reference directions using
        :meth:`pymoo.factory.get_reference_directions` and based on the mandatory
        argument ``ref_dirs_name``.

        See `Reference Directions <https://pymoo.org/misc/reference_directions.html>`_

        Args:
            ref_dirs_name: The reference directions name.
            **ref_dirs_settings: The reference directions settings.

        Returns:
            The reference directions. If ``ref_dirs_name`` is unknown,
                ``Riesz s-Energy`` is returned.

        Raises:
            ValueError: If multiple partitions are provided
                for a single-objective problem.
        """
        n_obj = self._problem.objective.dim

        if ref_dirs_name == "das-dennis":
            n_partitions = ref_dirs_settings.pop("n_partitions")
            return get_reference_directions(
                ref_dirs_name, n_obj, n_partitions=n_partitions
            )

        if ref_dirs_name == "multi-layer":
            n_partitions = ref_dirs_settings.pop("n_partitions")
            scaling_1 = ref_dirs_settings.pop("scaling_1")
            scaling_2 = ref_dirs_settings.pop("scaling_2")
            return get_reference_directions(
                ref_dirs_name,
                get_reference_directions(
                    "das-dennis", n_obj, n_partitions=n_partitions, scaling=scaling_1
                ),
                get_reference_directions(
                    "das-dennis", n_obj, n_partitions=n_partitions, scaling=scaling_2
                ),
            )

        if ref_dirs_name == "layer-energy":
            partitions = ref_dirs_settings.pop("partitions")
            if n_obj == 1 and np_size(partitions) > 1:
                msg = (
                    "For a single-objective problem, "
                    "the partitions array must be of size 1!"
                )
                raise ValueError(msg)
            return get_reference_directions(ref_dirs_name, n_obj, partitions)

        # By default, return Riesz s-Energy (in case of an unknown name is provided).
        n_points = ref_dirs_settings.pop("n_points")
        seed = ref_dirs_settings.pop("seed")
        return get_reference_directions(ref_dirs_name, n_obj, n_points, seed=seed)

    def _pre_run(self, problem: OptimizationProblem) -> None:
        """Take into account a new check for multi-objectives handling.

        Args:
            problem: The problem to be solved.
        """
        super()._pre_run(problem)
        self._check_mo_handling(problem)
        self._stop_crit_n_hv = self._settings.stop_crit_n_hv

    def _run(self, problem: OptimizationProblem) -> tuple[str, Any]:
        """Run the algorithm.

        Returns:
            The optimization result.

        Raises:
            ValueError: If the algorithm's name is not valid.
        """
        settings_ = self._settings.model_dump()
        # Instantiate the pymoo Problem.
        pymoo_problem_settings = {
            self.N_PROCESSES: settings_.pop(self.N_PROCESSES, 1),
            self.MAX_GEN: settings_.pop(self.MAX_GEN),
            self.HV_TOL_REL: settings_.pop(self.HV_TOL_REL),
            self.HV_TOL_ABS: settings_.pop(self.HV_TOL_ABS),
            self.STOP_CRIT_N_HV: settings_.pop(self.STOP_CRIT_N_HV),
        }
        pymoo_problem = PymooProblem(
            problem,
            self._settings.normalize_design_space,
            self,
            **pymoo_problem_settings,
        )

        # Problem type (continuous, discrete, mixed).
        type_var_unique = np_unique(pymoo_problem.data["type_var"])

        # Termination criterion based on the size of the design space.
        # It is required to avoid being stuck when dealing with discrete variables,
        # because GEMSEO does not count iterations already stored in database.
        if len(type_var_unique) == 1 and type_var_unique[0] == "integer":
            lower_bound, upper_bound = pymoo_problem.bounds()
            self._ds_size = int(np_prod(upper_bound - lower_bound + 1))
            if self._ds_size < problem.evaluation_counter.maximum:
                problem.add_listener(self._check_design_space_exploration)
        else:
            self._ds_size = inf

        # Filter settings to get only the ones of the global optimizer
        settings = self._filter_settings(settings_, BaseOptimizerSettings)

        evol_operators = {}
        for operator_name, operator_class in self.EVOLUTIONARY_OPERATORS.items():
            operator_instance = settings.pop(operator_name)
            if operator_instance is not None:
                self._check_operator_suitable(
                    pymoo_problem.xl,
                    pymoo_problem.xu,
                    operator_instance,
                    operator_class,
                )

                evol_operators[operator_name] = operator_instance
            elif (
                operator_name != self.SELECTION_OPERATOR
                and "integer" in type_var_unique
            ):
                msg = (
                    "Pymoo's default %s operator may not be suitable "
                    "for integer variables."
                )
                LOGGER.warning(msg, operator_name)

        # Common GeneticAlgorithm parameters.
        common_settings = {
            "eliminate_duplicates": settings.pop("eliminate_duplicates"),
            "n_offsprings": settings.pop("n_offsprings"),
        }

        algo_name = (
            self.__PYMOO_PREFIX
            + self.ALGORITHM_INFOS[self._algo_name].internal_algorithm_name
        )
        if algo_name == self.PYMOO_RNSGA3:
            algorithm = RNSGA3(
                ref_points=settings.pop("ref_points"),
                pop_per_ref_point=settings.pop("pop_per_ref_point"),
                mu=settings.pop("mu"),
                **evol_operators,
                **common_settings,
            )
        elif algo_name == self.PYMOO_UNSGA3:
            ref_dirs_name = settings.pop("ref_dirs_name")
            directions = self._get_ref_dirs(ref_dirs_name, **settings)
            algorithm = UNSGA3(
                ref_dirs=directions,
                **evol_operators,
                **common_settings,
            )
        elif algo_name == self.PYMOO_NSGA3:
            ref_dirs_name = settings.pop("ref_dirs_name")
            directions = self._get_ref_dirs(ref_dirs_name, **settings)
            algorithm = NSGA3(
                ref_dirs=directions,
                **evol_operators,
                **common_settings,
            )
        elif algo_name == self.PYMOO_NSGA2:
            algorithm = NSGA2(
                pop_size=settings.pop("pop_size"),
                **evol_operators,
                **common_settings,
            )
        elif algo_name == self.PYMOO_GA:
            algorithm = GA(
                pop_size=settings.pop("pop_size"),
                **evol_operators,
                **common_settings,
            )
        else:  # pragma: no cover
            # GEMSEO will check in advance if the algorithm is supported.
            msg = (
                f"Algorithm not supported: "
                f"{self.ALGORITHM_INFOS[self._algo_name].internal_algorithm_name}"
            )
            raise ValueError(msg)

        res = minimize(
            pymoo_problem,
            algorithm=algorithm,
            termination=("n_gen", self.pymoo_n_gen),
            return_least_infeasible=True,
            **settings,
        )

        return res.message, res.success

    def _get_optimization_result(
        self,
        problem: OptimizationProblem,
        message: str | None = None,
        status: int | None = None,
    ) -> OptimizationResult | MultiObjectiveOptimizationResult:
        """Return the optimization result adapted to the dimension of the problem.

        Return an
        :class:`~gemseo.algos.opt_result.OptimizationResult` instance adapted for
        multi-objective results (see
        :class:`~gemseo_pymoo.algos.opt_result_mo.MultiObjectiveOptimizationResult`).

        Args:
            problem: The problem to be solved.
            message: The message associated with the termination criterion.
            status: The status associated with the termination criterion.

        Returns:
            An optimization result object based on the optimum found.
        """
        # Single-objective problem.
        if problem.objective.dim == 1:
            return OptimizationResult.from_optimization_problem(
                problem,
                message=message,
                status=status,
                optimizer_name=self.algo_name,
            )
        return MultiObjectiveOptimizationResult.from_optimization_problem(
            problem, message=message, status=status, optimizer_name=self.algo_name
        )

    def _check_design_space_exploration(self, design_variables: ndarray) -> None:
        """Check on the design space exploration.

        Args:
            design_variables: The design variables vector.

        Raises:
            DesignSpaceExploredException: If the design space
                has been completely explored.
        """
        if self._problem.evaluation_counter.current == self._ds_size:
            raise DesignSpaceExploredException

    def _get_early_stopping_result(
        self, problem: OptimizationProblem, termination_criterion: TerminationCriterion
    ) -> OptimizationResult | MultiObjectiveOptimizationResult:
        """Retrieve the best known iterate when max iter has been reached.

        Takes into account some termination criteria specific for multi-objective
        and/or mixed variables problems:

            - :class:`~gemseo_pymoo.algos.stop_criteria.DesignSpaceExploredException`
            - :class:`~gemseo_pymoo.algos.stop_criteria.MaxGenerationsReached`.

        Args:
            problem: The problem to be solved.
            termination_criterion: A termination criterion.

        Returns:
            An optimization result object.
        """
        if isinstance(termination_criterion, DesignSpaceExploredException):
            message = (
                f"All {self._ds_size} points of the design space have been explored. "
                f"GEMSEO stopped the driver."
            )
            return self._get_optimization_result(problem, message)

        if isinstance(termination_criterion, MaxGenerationsReached):
            message = (
                "Maximum number of generations reached. GEMSEO stopped the driver."
            )
            return self._get_optimization_result(problem, message)

        if isinstance(termination_criterion, HyperVolumeToleranceReached):
            message = (
                f"{self._stop_crit_n_hv} successive iterates of the hypervolume "
                "indicator are closer than hv_tol_rel or hv_tol_abs. "
                "GEMSEO stopped the driver."
            )
            return self._get_optimization_result(problem, message)
        return super()._get_early_stopping_result(problem, termination_criterion)

    def _log_result(
        self, problem: OptimizationProblem, max_design_space_dimension_to_log: int
    ) -> None:
        if problem.objective.dim == 1:
            super()._log_result(problem, max_design_space_dimension_to_log)
        else:
            LOGGER.info("%s", problem.solution)
