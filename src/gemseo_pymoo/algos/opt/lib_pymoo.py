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
from typing import Any
from typing import Callable
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

from gemseo.algos.opt.opt_lib import OptimizationAlgorithmDescription
from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.algos.stop_criteria import TerminationCriterion
from gemseo.utils.python_compatibility import Final
from numpy import inf
from numpy import ndarray
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
from pymoo.core.population import Population
from pymoo.core.sampling import Sampling
from pymoo.core.selection import Selection
from pymoo.factory import get_crossover
from pymoo.factory import get_mutation
from pymoo.factory import get_reference_directions
from pymoo.factory import get_sampling
from pymoo.factory import get_selection
from pymoo.operators.mixed_variable_operator import MixedVariableCrossover
from pymoo.operators.mixed_variable_operator import MixedVariableMutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling
from pymoo.optimize import minimize
from pymoo.util.reference_direction import MultiLayerReferenceDirectionFactory
from pymoo.util.reference_direction import ReferenceDirectionFactory

from gemseo_pymoo.algos.opt.core.pymoo_problem_adapater import PymooProblem
from gemseo_pymoo.algos.opt_result_mo import MultiObjectiveOptimizationResult
from gemseo_pymoo.algos.opt_result_mo import Pareto
from gemseo_pymoo.algos.stop_criteria import DesignSpaceExploredException
from gemseo_pymoo.algos.stop_criteria import HyperVolumeToleranceReached
from gemseo_pymoo.algos.stop_criteria import MaxGenerationsReached

LOGGER = logging.getLogger(__name__)

EvolutionaryOperatorTypes = Union[Crossover, Mutation, Sampling, Selection]
EvolutionaryOperatorSimpleOptionsType = Tuple[str, Dict[str, Union[float, bool]]]
EvolutionaryOperatorOptionsType = Union[
    str,  # Operator's name.
    EvolutionaryOperatorSimpleOptionsType,  # Operator's name with options.
    Dict[
        str,  # Variable type.
        Union[str, EvolutionaryOperatorTypes, EvolutionaryOperatorSimpleOptionsType],
    ],  # Mixed variables.
]
EvolutionaryMixedOperatorTypes = Union[
    MixedVariableCrossover, MixedVariableMutation, MixedVariableSampling, None
]


@dataclass
class PymooAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of an optimization algorithm from the pymoo library."""

    library_name: str = "pymoo"

    handle_integer_variables: bool = True

    require_grad: bool = False

    handle_equality_constraints: bool = False

    handle_inequality_constraints: bool = True

    positive_constraints: bool = False

    problem_type: str = OptimizationProblem.NON_LINEAR_PB


class PymooOpt(OptimizationLibrary):
    """Pymoo optimization library interface.

    See :class:`gemseo.algos.opt.opt_lib.OptimizationLibrary`.
    """

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

    __PYMOO_WEBPAGE: Final[str] = "https://www.pymoo.org"
    """The pymoo webpage."""

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

    EVOLUTIONARY_OPERATORS: Final[list[str]] = [
        CROSSOVER_OPERATOR,
        MUTATION_OPERATOR,
        SAMPLING_OPERATOR,
        SELECTION_OPERATOR,
    ]
    """A list with all evolutionary operators available."""

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

    __PYMOO_METADATA: Final[dict[str, tuple[str, str]]] = {
        PYMOO_GA: ("Genetic Algorithm", "soo/nonconvex/ga.html#nb-ga"),
        PYMOO_NSGA2: (
            "Non-dominated Sorting Genetic Algorithm II",
            "moo/nsga2.html#nb-nsga2",
        ),
        PYMOO_NSGA3: (
            "Non-dominated Sorting Genetic Algorithm III",
            "moo/nsga3.html#nb-nsga3",
        ),
        PYMOO_UNSGA3: ("Unified NSGA3", "moo/unsga3.html#nb-unsga3"),
        PYMOO_RNSGA3: ("Reference Point Based NSGA3", "moo/rnsga3.html#nb-rnsga3"),
    }
    """The description and webpage link of the pymoo algorithms."""

    LIBRARY_NAME: Final[str] = "pymoo"
    """The library's name."""

    pymoo_n_gen: int = 10000000
    """The pymoo's termination criterion based on the number of generations."""

    _stop_crit_n_hv: int = 5
    """The number of generations to account for in the hypervolume convergence check."""

    _ds_size: int
    """The design space size."""

    _operators_map: dict[
        str,
        tuple[
            Callable[[str, Any], EvolutionaryOperatorTypes],
            Callable[
                [Sequence[str], dict[str, EvolutionaryOperatorTypes]],
                EvolutionaryMixedOperatorTypes,
            ],
            EvolutionaryOperatorTypes | Sampling | Population | ndarray,
        ],
    ]
    """Map the operator's nature to the corresponding pymoo's getters and classes."""

    def __init__(self) -> None:
        """Constructor.

        Generate the library dict, which contains the list
        of algorithms with their characteristics:

        - does it require gradient
        - does it handle equality constraints
        - does it handle inequality constraints
        """
        super().__init__()

        # The design space size (useful for finite discrete problems).
        self._ds_size = inf

        # See https://www.pymoo.org/misc/constraints.html for eq constraints.
        for algo_name, algo_value in self.__PYMOO_METADATA.items():
            internal_name = algo_name.replace(self.__PYMOO_, "")
            self.descriptions[algo_name] = PymooAlgorithmDescription(
                algorithm_name=internal_name,
                internal_algorithm_name=internal_name,
                description=algo_value[0],
                website=f"{self.__PYMOO_WEBPAGE}/algorithms/{algo_value[1]}",
                handle_multiobjective=not (internal_name == "GA"),
            )

        self._operators_map = {
            self.SAMPLING_OPERATOR: (
                get_sampling,
                MixedVariableSampling,
                (Sampling, Population, ndarray),
            ),
            self.MUTATION_OPERATOR: (get_mutation, MixedVariableMutation, Mutation),
            self.CROSSOVER_OPERATOR: (get_crossover, MixedVariableCrossover, Crossover),
            self.SELECTION_OPERATOR: (get_selection, None, Selection),
        }

    def _check_mo_handling(
        self,
        algo_name: str,
        opt_problem: OptimizationProblem,
    ) -> None:
        """Check if the algorithm is capable of handling the optimization problem.

        All algorithms implemented are capable of handling  multi-objective problems,
        except the :class:`~pymoo.algorithms.soo.nonconvex.ga.GA` one.

        Args:
            algo_name: The name of the algorithm.
            opt_problem: The problem to be solved.

        Raises:
            ValueError: If the algo cannot handle the problem to be solved.
        """
        if (
            opt_problem.objective.dim > 1
            and not self.descriptions[algo_name].handle_multiobjective
        ):
            raise ValueError(
                f"Requested optimization algorithm {self.algo_name} can not handle "
                "multiple objectives."
            )

    def _get_options(
        self,
        max_iter: int = 999,
        max_gen: int = 10000000,
        ftol_rel: float = 1e-9,
        ftol_abs: float = 1e-9,
        xtol_rel: float = 1e-9,
        xtol_abs: float = 1e-9,
        hv_tol_rel: float = 1e-9,
        hv_tol_abs: float = 1e-9,
        stop_crit_n_x: int = 3,
        stop_crit_n_hv: int = 5,
        normalize_design_space: bool = True,
        eq_tolerance: float = 1e-2,
        ineq_tolerance: float = 1e-4,
        ref_dirs: ndarray | None = None,
        pop_size: int = 100,
        sampling: Sampling | Population | EvolutionaryOperatorOptionsType | None = None,
        selection: Selection | EvolutionaryOperatorOptionsType | None = None,
        mutation: Mutation | EvolutionaryOperatorOptionsType | None = None,
        crossover: Crossover | EvolutionaryOperatorOptionsType | None = None,
        eliminate_duplicates: bool = True,
        n_offsprings: int | None = None,
        seed: int = 1,
        pop_per_ref_point: int = 1,
        mu: float = 0.1,
        ref_points: ndarray | None = None,
        n_partitions: int = 20,
        n_points: int | None = None,
        partitions: ndarray | None = None,
        scaling_1: float | None = None,
        scaling_2: float | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        r"""Set the options default values.

        To get the best and up-to-date information about algorithms options, go to
        pymoo's algorithms `documentation <https://pymoo.org/algorithms/index.html>`_

        Args:
            max_iter: The maximum number of iterations, i.e. unique calls to f(x).
            max_gen: The maximum number of generations.
            ftol_rel: A stop criterion, the relative tolerance on the objective
                function. If abs(f(xk)-f(xk+1))/abs(f(xk))<= ftol_rel: stop.
            ftol_abs: A stop criterion, the absolute tolerance on the objective
                function. If abs(f(xk)-f(xk+1))<= ftol_rel: stop.
            xtol_rel: A stop criterion, the relative tolerance on the design variables.
                If norm(xk-xk+1)/norm(xk)<= xtol_rel: stop.
            xtol_abs: A stop criterion, absolute tolerance on the design variables.
                If norm(xk-xk+1)<= xtol_abs: stop.
            hv_tol_rel: A stop criterion, the relative tolerance on the hypervolume
                convergence check. If norm(xk-xk+1)/norm(xk)<= hv_tol_rel: stop.
            hv_tol_abs: A stop criterion, absolute tolerance on the hypervolume
                convergence check. If norm(xk-xk+1)<= hv_tol_abs: stop.
            stop_crit_n_x: The number of design vectors to account for during the
               criteria check.
            stop_crit_n_hv: The number of generations to account for during the
               criterion check on the hypervolume indicator.
            normalize_design_space: If True, scale the variables to the range [0, 1].
            eq_tolerance: The equality tolerance.
            ineq_tolerance: The inequality tolerance.
            ref_dirs: The reference directions.
            pop_size: The population size.
            sampling: The sampling process that generates the initial population.
                If None, the algorithm's default is used.
            selection: The mating selection operator.
                If None, the algorithm's default is used.
            mutation: The mutation operator.
                If None, the algorithm's default is used.
            crossover: The crossover operator used to create offsprings.
                If None, the algorithm's default is used.
            eliminate_duplicates: If True, eliminate duplicates after merging
                the parent and the offspring population.
            n_offsprings: Number of offspring that are created through mating.
                If None, it will be set equal to the population size.
            seed: The random seed to be used.
            pop_per_ref_point: The size of the population used for each reference point.
            mu: The scaling of the reference lines used during survival selection.
                Increasing mu will generate solutions with a larger spread.
            ref_points: The reference points (Aspiration Points) as a NumPy array
                where each row represents a point and each column a variable.
            n_partitions: The number of gaps between two consecutive points
                along an objective axis.
            n_points: The number of points on the unit simplex.
            partitions: The custom partitions.
            scaling_1: The scaling of the first simplex.
            scaling_2: The scaling of the second simplex.
            **options: The other algorithm options.

        Notes:
            The pymoo library allows the user to define custom operators
            to manage the processes of sampling, crossover, mutation and selection.

            In GEMSEO, these operators can be provided in two ways:

                1. Operator's name along with associated options:
                    In such case the associated ``getter``
                    (e.g. :meth:`pymoo.factory.get_sampling`) will be used to
                    instantiate the operator.
                2. Custom operator:
                    This is done by providing a dictionary with the key ``custom``
                    and the custom operator instance as its value.

            For the special case of mixed variables (discrete and continous),
            a dictionnary containing each variable type as keys and
            their associated operators as values is expected.

            In case no info regarding these operators is provided, pymoo's
            default will be used. Nevertheless, be aware that they may not be
            suitable for problems containing integer design variables.

            For details about each operator's options,
            refer to https://pymoo.org/operators/index.html

        Example:
            Let us consider an optimization problem where
            all the design variables are discrete (integers).
            The following operators could be provided among the options:

            >>> class MyIntegerMutationOperator(Mutation):
            ...
            ...     def _do(self, problem, x, **kwargs_):
            ...         # my integer mutation strategy
            ...
            ...
            >>> operators = dict(
            ...     selection='random',
            ...     sampling='int_random',
            ...     crossover=('int_sbx', dict(prob=0.9, eta=30)),
            ...     mutation=dict(custom=MyIntegerMutationOperator())
            ... )

            Now, if we consider an optimization problem with
            mixed design variables (discrete and continuous).
            The following operators could be provided among the options:

            >>> class MyFloatMutationOperator(Mutation):
            ...
            ...     def _do(self, problem, x, **kwargs_):
            ...         # my float mutation strategy
            ...
            ...
            >>> operators = dict(
            ...     selection='random',
            ...     sampling=dict(integer='int_random', float='real_random'),
            ...     crossover=dict(
            ...         integer=('int_sbx', dict(prob=0.9, eta=30)),
            ...         float=('real_sbx', dict(prob=0.9, eta=30))
            ...     ),
            ...     mutation=dict(
            ...         integer=dict(custom=MyIntegerMutationOperator()),
            ...         float=dict(custom=MyFloatMutationOperator())
            ...     )
            ... )

            Note that in this case, we must provide an operator for each variable type.
            Moreover, the selection operator is not concerned in the case of
            mixed variables.
        """
        popts = self._process_options(
            max_iter=max_iter,
            max_gen=max_gen,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            hv_tol_rel=hv_tol_rel,
            hv_tol_abs=hv_tol_abs,
            stop_crit_n_x=stop_crit_n_x,
            stop_crit_n_hv=stop_crit_n_hv,
            normalize_design_space=normalize_design_space,
            ineq_tolerance=ineq_tolerance,
            eq_tolerance=eq_tolerance,
            selection=selection,
            sampling=sampling,
            mutation=mutation,
            crossover=crossover,
            eliminate_duplicates=eliminate_duplicates,
            n_offsprings=n_offsprings,
            seed=seed,
            mu=mu,
            ref_points=ref_points,
            pop_per_ref_point=pop_per_ref_point,
            pop_size=pop_size,
            ref_dirs=ref_dirs,
            n_partitions=n_partitions,
            n_points=n_points,
            partitions=partitions,
            scaling_1=scaling_1,
            scaling_2=scaling_2,
            **options,
        )
        return popts

    @staticmethod
    def _check_operator_suitable(
        type_vars: str, lower_bounds: ndarray, upper_bounds: ndarray, operator_name: str
    ) -> None:
        """Check operator suitability according to design variables type and bounds.

        Args:
            type_vars: The design variables' type (``integer`` of ``float``).
            lower_bounds: The design variables' lower bounds.
            upper_bounds: The design variables' upper bounds.
            operator_name: The pymoo operator's name.

        Raises:
            ValueError: If ``operator_name`` refers to the mutation operator
                :class:`~pymoo.operators.mutation.pm.PolynomialMutation` and at least
                one design variable has equal lower and upper bounds.
        """
        suitability_tags = {
            "integer": ["int", "bin", "none"],
            "float": ["real", "perm"],
        }
        if all(tag not in operator_name for tag in suitability_tags[type_vars]):
            msg = "The %s operator may not be suitable for %s variables"
            LOGGER.warning(msg, operator_name, type_vars)

        # Anticipate 'division by zero' errors when using PolynomialMutation,
        # which occurs when we have equal lower and upper bounds.
        if "pm" in operator_name and any(upper_bounds == lower_bounds):
            raise ValueError(
                "PolynomialMutation cannot handle equal lower and upper bounds!\n"
                "Consider setting those design variables as constants of your problem."
            )

    @staticmethod
    def _parse_cls_options(
        pymoo_cls_options: EvolutionaryOperatorOptionsType,
    ) -> tuple[str, Sequence[float], dict[str, float | bool]]:
        """Parse the evolutionary operator's options.

        Args:
            pymoo_cls_options: The options to be parsed.

        Returns:
            The arguments ready to be passed to the corresponding operator's ``getter``
                from :mod:`pymoo.factory`.
        """
        if isinstance(pymoo_cls_options, str):
            return pymoo_cls_options, [], {}

        pymoo_cls_options = list(pymoo_cls_options)
        name = pymoo_cls_options.pop(0)
        args = []
        kwargs = {}
        while pymoo_cls_options:
            if isinstance(pymoo_cls_options[0], dict):
                kwargs.update(pymoo_cls_options.pop(0))
            else:
                args.extend(pymoo_cls_options.pop(0))

        return name, args, kwargs

    def _get_operator(
        self,
        types: ndarray | Sequence[str],
        lower_bounds: ndarray,
        upper_bounds: ndarray,
        nature: str,
        operator_options: EvolutionaryOperatorOptionsType,
    ) -> EvolutionaryOperatorTypes:
        """Instantiate an evolutionary operator based on the design variables type.

        Args:
            types: The design variables type.
            lower_bounds: The design variables lower bounds.
            upper_bounds: The design variables upper bounds.
            nature: The operator's nature
                ("sampling", "crossover", "mutation", "selection").
            operator_options: The operator's custom options.

        Returns:
            Instance of a pymoo's evolutionary operator.

        Raises:
            TypeError: If a custom operator is provided,
                but it is not appropriate considered its nature.
        """
        get_operator, mixed_operator_class, operator_class = self._operators_map[nature]

        # A custom operator instance is provided.
        if "custom" in operator_options:
            custom_instance = operator_options["custom"]
            if not isinstance(custom_instance, operator_class):
                raise TypeError(
                    f"{custom_instance} must be an instance of {operator_class} "
                    "or inherit from it!"
                )
            return custom_instance

        # Use the pymoo factory getter to instantiate
        # an operator with the options provided.
        if len(set(types)) == 1 or nature == self.SELECTION_OPERATOR:
            cls_options = self._parse_cls_options(operator_options)
            self._check_operator_suitable(
                types[0], lower_bounds, upper_bounds, cls_options[0]
            )
            return get_operator(cls_options[0], *cls_options[1], **cls_options[2])

        f_idx = (types == "float").nonzero()[0]
        i_idx = (types == "int").nonzero()[0]
        return mixed_operator_class(
            types,
            dict(
                float=self._get_operator(
                    ["float"],
                    lower_bounds[f_idx],
                    upper_bounds[f_idx],
                    nature,
                    operator_options.get("float"),
                ),
                integer=self._get_operator(
                    ["integer"],
                    lower_bounds[i_idx],
                    upper_bounds[i_idx],
                    nature,
                    operator_options.get("int", operator_options.get("integer")),
                ),
            ),
        )

    def _get_ref_dirs(
        self,
        ref_dirs_name: str,
        **ref_dirs_options: dict[str, str],
    ) -> ReferenceDirectionFactory | MultiLayerReferenceDirectionFactory:
        r"""Return the reference directions.

        Get the reference directions using
        :meth:`pymoo.factory.get_reference_directions` and based on the mandatory
        argument ``ref_dirs_name``.

        See `Reference Directions <https://pymoo.org/misc/reference_directions.html>`_

        Args:
            ref_dirs_name: The reference directions name.
            **ref_dirs_options: The reference directions options.

        Returns:
            The reference directions. If ``ref_dirs_name`` is unknown,
                ``Riesz s-Energy`` is returned.

        Raises:
            ValueError: If multiple partitions are provided
                for a single-objective problem.
        """
        n_obj = self.problem.objective.dim

        if ref_dirs_name == "das-dennis":
            n_partitions = ref_dirs_options.pop("n_partitions")
            return get_reference_directions(
                ref_dirs_name, n_obj, n_partitions=n_partitions
            )

        if ref_dirs_name == "multi-layer":
            n_partitions = ref_dirs_options.pop("n_partitions")
            scaling_1 = ref_dirs_options.pop("scaling_1")
            scaling_2 = ref_dirs_options.pop("scaling_2")
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
            partitions = ref_dirs_options.pop("partitions")
            if n_obj == 1 and np_size(partitions) > 1:
                raise ValueError(
                    "For a single-objective problem, "
                    "the partitions array must be of size 1!"
                )
            return get_reference_directions(ref_dirs_name, n_obj, partitions)

        # By default, return Riesz s-Energy (in case of an unknown name is provided).
        n_points = ref_dirs_options.pop("n_points")
        seed = ref_dirs_options.pop("seed")
        return get_reference_directions(ref_dirs_name, n_obj, n_points, seed=seed)

    def _pre_run(
        self, problem: OptimizationProblem, algo_name: str, **options: Any
    ) -> None:
        """Take into account a new check for multi-objectives handling.

        Args:
            problem: The problem to be solved.
            algo_name: The name of the algorithm.
            **options: The options for the algorithm, see associated JSON file.
        """
        super()._pre_run(problem, algo_name, **options)
        self._check_mo_handling(algo_name, problem)
        self._stop_crit_n_hv = options.get(self.STOP_CRIT_N_HV)

    def _run(
        self, **options: Any
    ) -> OptimizationResult | MultiObjectiveOptimizationResult:
        """Run the algorithm.

        Args:
            **options: The options for the algorithm.

        Returns:
            The optimization result.

        Raises:
            ValueError: If the algorithm's name is not valid.
        """
        # Remove normalization from algorithm's options.
        normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)

        # Instantiate the pymoo Problem.
        pymoo_problem_options = {
            self.N_PROCESSES: options.pop(self.N_PROCESSES, 1),
            self.MAX_GEN: options.pop(self.MAX_GEN),
            self.HV_TOL_REL: options.pop(self.HV_TOL_REL),
            self.HV_TOL_ABS: options.pop(self.HV_TOL_ABS),
            self.STOP_CRIT_N_HV: options.pop(self.STOP_CRIT_N_HV),
        }
        pymoo_problem = PymooProblem(
            self.problem, normalize_ds, self, **pymoo_problem_options
        )

        # Problem type (continuous, discrete, mixed).
        type_var_unique = np_unique(pymoo_problem.data["type_var"])

        # Termination criterion based on the size of the design space.
        # It is required to avoid being stuck when dealing with discrete variables,
        # because GEMSEO does not count iterations already stored in database.
        if len(type_var_unique) == 1 and type_var_unique[0] == "integer":
            l_b, u_b = pymoo_problem.bounds()
            self._ds_size = int(np_prod(u_b - l_b + 1))
            if self._ds_size < self.problem.max_iter:
                self.problem.add_callback(self._check_design_space_exploration)
        else:
            self._ds_size = inf

        # Instantiate evolutionary operators.
        evol_operators = {}
        for operator in self.EVOLUTIONARY_OPERATORS:
            opts = options.pop(operator)
            if opts is not None:
                evol_operators[operator] = self._get_operator(
                    pymoo_problem.data["type_var"],
                    pymoo_problem.xl,
                    pymoo_problem.xu,
                    operator,
                    opts,
                )
            elif operator != self.SELECTION_OPERATOR and "integer" in type_var_unique:
                msg = (
                    "Pymoo's default %s operator may not be suitable "
                    "for integer variables!"
                )
                LOGGER.warning(msg, operator)

        # Common GeneticAlgorithm parameters.
        common_options = {
            "eliminate_duplicates": options.pop("eliminate_duplicates"),
            "n_offsprings": options.pop("n_offsprings"),
        }

        algo_name = self.__PYMOO_PREFIX + self.internal_algo_name
        if algo_name == self.PYMOO_RNSGA3:
            algorithm = RNSGA3(
                ref_points=options.pop("ref_points"),
                pop_per_ref_point=options.pop("pop_per_ref_point"),
                mu=options.pop("mu"),
                **evol_operators,
                **common_options,
            )
        elif algo_name == self.PYMOO_UNSGA3:
            ref_dirs_name = options.pop("ref_dirs_name")
            directions = self._get_ref_dirs(ref_dirs_name, **options)
            algorithm = UNSGA3(
                ref_dirs=directions,
                **evol_operators,
                **common_options,
            )
        elif algo_name == self.PYMOO_NSGA3:
            ref_dirs_name = options.pop("ref_dirs_name")
            directions = self._get_ref_dirs(ref_dirs_name, **options)
            algorithm = NSGA3(
                ref_dirs=directions,
                **evol_operators,
                **common_options,
            )
        elif algo_name == self.PYMOO_NSGA2:
            algorithm = NSGA2(
                pop_size=options.pop("pop_size"),
                **evol_operators,
                **common_options,
            )
        elif algo_name == self.PYMOO_GA:
            algorithm = GA(
                pop_size=options.pop("pop_size"),
                **evol_operators,
                **common_options,
            )
        else:  # pragma: no cover
            # GEMSEO will check in advance if the algorithm is supported.
            raise ValueError(f"Algorithm not supported : {self.internal_algo_name}")

        res = minimize(
            pymoo_problem,
            algorithm=algorithm,
            termination=("n_gen", self.pymoo_n_gen),
            return_least_infeasible=True,
            **options,
        )

        return self.get_optimum_from_database(res.message, res.success)

    def _post_run(
        self,
        problem: OptimizationProblem,
        algo_name: str,
        result: OptimizationResult | MultiObjectiveOptimizationResult,
        **options: Any,
    ) -> None:
        """Print a design space suitable for multi-objective problems.

        Args:
            problem: The problem to be solved.
            algo_name: The name of the algorithm.
            result: The optimization result.
            **options: The options for the algorithm, see associated JSON file.
        """
        if self.problem.objective.dim == 1:
            super()._post_run(problem, algo_name, result, **options)
        else:
            LOGGER.info("%s", result)
            problem.solution = result

    def get_optimum_from_database(
        self, message: str | None = None, status: int | None = None
    ) -> OptimizationResult | MultiObjectiveOptimizationResult:
        """Retrieve the optimum from the database.

        Override the super class method in order to return an
        :class:`~gemseo.algos.opt_result.OptimizationResult` instance adapted for
        multi-objective results (see
        :class:`~gemseo_pymoo.algos.opt_result_mo.MultiObjectiveOptimizationResult`).

        Args:
            message: The message from the optimizer.
            status: The status from the optimizer.

        Returns:
            An optimization result object based on the optimum found.
        """
        # Single-objective problem.
        if self.problem.objective.dim == 1:
            return super().get_optimum_from_database(message, status)

        if len(self.problem.database) == 0:
            return MultiObjectiveOptimizationResult(
                optimizer_name=self.algo_name,
                message=message,
                status=status,
                n_obj_call=0,
            )
        x_0 = self.problem.database.get_x_by_iter(0)
        # Compute the best feasible or infeasible point.
        f_opt, x_opt, is_feas, c_opt, c_opt_grad = self.problem.get_optimum()

        if f_opt is not None and not self.problem.minimize_objective:
            f_opt = -f_opt

        # There are no pareto efficient solutions for unfeasible problems.
        pareto = Pareto(self.problem) if is_feas else None

        return MultiObjectiveOptimizationResult(
            x_0=x_0,
            x_opt=x_opt,
            f_opt=f_opt,  # objective vector norm
            optimizer_name=self.algo_name,
            message=message,
            status=status,
            n_obj_call=self.problem.objective.n_calls,
            is_feasible=is_feas,
            constraints_values=c_opt,
            constraints_grad=c_opt_grad,
            pareto=pareto,
        )

    def _check_design_space_exploration(self, design_variables: ndarray) -> None:
        """Check on the design space exploration.

        Args:
            design_variables: The design variables vector.

        Raises:
            DesignSpaceExploredException: If the design space
                has been completely explored.
        """
        if self.problem.current_iter == self._ds_size:
            raise DesignSpaceExploredException

    def _termination_criterion_raised(
        self, error: TerminationCriterion
    ) -> OptimizationResult | MultiObjectiveOptimizationResult:
        """Retrieve the best known iterate when max iter has been reached.

        Takes into account some termination criteria specific for multi-objective
        and/or mixed variables problems:

            - :class:`~gemseo_pymoo.algos.stop_criteria.DesignSpaceExploredException`
            - :class:`~gemseo_pymoo.algos.stop_criteria.MaxGenerationsReached`.

        Args:
            error: The obtained error from the algorithm.

        Returns:
            An optimization result object.
        """
        if isinstance(error, DesignSpaceExploredException):
            message = (
                f"All {self._ds_size} points of the design space have been explored. "
                f"GEMSEO stopped the driver."
            )
            return self.get_optimum_from_database(message)

        if isinstance(error, MaxGenerationsReached):
            message = (
                "Maximum number of generations reached. GEMSEO stopped the driver."
            )
            return self.get_optimum_from_database(message)

        if isinstance(error, HyperVolumeToleranceReached):
            message = (
                f"{self._stop_crit_n_hv} successive iterates of the hypervolume "
                "indicator are closer than hv_tol_rel or hv_tol_abs. "
                "GEMSEO stopped the driver."
            )
            return self.get_optimum_from_database(message)

        return super()._termination_criterion_raised(error)
