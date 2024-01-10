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
r"""**Knapsack problem**.

This module implements the Knapsack problem.

In its simplest form, it states that:

    *Given a set of items, each with a given weight and value,
    determine the number of each item to include in a collection
    so that the total weight is less than or equal to a given weight capacity
    and the total value is as large as possible.*

.. math::

   \begin{aligned}
   \text{maximize the total knapsack value } & \sum_{i=1}^{n} value_i * x_i \\
   \text{with respect to the design variables }&x_i \\
   \text{subject to the general constraints }
   & \sum_{i=1}^{n} weight_i * x_i \leq capacity_weight\\
   & \sum_{i=1}^{n} x_i \leq capacity_items\\
   \text{subject to the search domain }
   & x_i \in \mathbb{N}
   \end{aligned}

Multiple variations of the Knapsack problem can be achieved
depending on the inputs provided.

Moreover, a multi-objective version of this problem is also available,
in which the following new objective function is added to previous formulation:

.. math::

   \begin{aligned}
   \text{minimize the number of items carried } & \sum_{i=1}^{n} x_i \\
   \end{aligned}
"""

from __future__ import annotations

import logging

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from numpy import atleast_1d
from numpy import ndarray
from numpy import ones
from numpy import sum as np_sum
from numpy import zeros
from numpy.random import default_rng

LOGGER = logging.getLogger(__name__)


class Knapsack(OptimizationProblem):
    """Generic knapsack optimization problem.

    Different `variations <https://en.wikipedia.org/wiki/List_of_knapsack_problems>`_
    can be achieved:

    - 0/1 or Binary Knapsack problem:

        Given a set of :math:`n` items, each with a weight :math:`w_i` and
        a value :math:`v_i`, and a knapsack with a maximum weight capacity :math:`W`.
        Choose which items to pack in order to maximize the total knapsack value
        while respecting its weight capacity.

    - Unbounded Knapsack problem:

        With respect to the Binary variant, it removes the restriction that there is
        only one of each item. This can be achieved by setting the attribute
        :attr:`.binary` to False, which will remove the upper bound of the
        design variables.

    - Bounded Knapsack problem:

        With respect to the Binary variant, it specifies an upper bound for each item.
        This can be achieved by providing an array :attr:`.items_ub` with the upper
        bound relative to each item.

    Moreover, an additional constraint regarding the total number of items can be added.
    This is achieved through the attribute :attr:`.capacity_items` and will limit
    the number of items that fit into the knapsack.
    """

    values: ndarray
    """The knapsack items' value."""

    weights: ndarray
    """The knapsack items' weight."""

    capacity_weight: float
    """The knapsack weight capacity."""

    capacity_items: int
    """The knapsack number of items capacity."""

    def __init__(
        self,
        values: ndarray,
        weights: ndarray,
        items_ub: ndarray | None = None,
        binary: bool = True,
        capacity_weight: float | None = None,
        capacity_items: int | None = None,
        initial_guess: ndarray | None = None,
    ) -> None:
        """The constructor.

        Initialize the Knapsack :class:`~gemseo.algos.opt_problem.OptimizationProblem`
        by defining the :class:`~gemseo.algos.design_space.DesignSpace` and the
        objective and constraint functions.

        The number of items in the problem is deduced from the :attr:`.values` array.

        Args:
            values: The items' values.
            weights: The items' weights.
            items_ub: The items' upper bounds.
                If None, an unlimited number of each item is allowed.
            binary: If True, the upper bound of design variables is set to 1.
            capacity_weight: The knapsack weight capacity.
                If None, the knapsack will have an unlimited weight capacity.
            capacity_items: The knapsack number of items capacity.
                If None, the knapsack will accept an unlimited total number of items.
            initial_guess: The initial guess for the optimal solution.
                If None, the initial guess will be an empty knapsack (0, 0, ..., 0).

        Raises:
            ValueError: Either if the provided arrays do not have the same length or
                if no capacity is provided.
        """
        # Number of items.
        n_items = len(values)

        if len(weights) != n_items:
            raise ValueError(
                "weights and values must have the same number of elements! "
                f"{len(weights)} != {n_items}"
            )

        # The knapsack must be constrained.
        if capacity_weight is None and capacity_items is None:
            raise ValueError("You have to provide at least one type of capacity!")

        if binary:
            if items_ub is None:
                # Binary variant.
                items_ub = ones(n_items)
            elif len(items_ub) != n_items:
                raise ValueError(
                    "items_ub and values must have the same number of elements! "
                    f"{len(items_ub)} != {n_items}"
                )
            else:
                LOGGER.warning(
                    "binary option is ignored because "
                    "the items upper bounds were provided!"
                )

        self.values = values
        self.weights = weights
        self.capacity_items = capacity_items
        self.capacity_weight = capacity_weight

        design_space = DesignSpace()
        design_space.add_variable(
            "x",
            size=n_items,
            l_b=0,
            u_b=items_ub,
            var_type=DesignSpace.DesignVariableType.INTEGER,
        )

        if initial_guess is None or len(initial_guess) == n_items:
            design_space.set_current_value(zeros(n_items))
        else:
            raise ValueError(f"initial_guess must have {n_items} elements!")

        super().__init__(design_space)

        self.objective = MDOFunction(
            self.compute_knapsack_value,
            name="knapsack",
            f_type=MDOFunction.FunctionType.OBJ,
            expr="sum(values * x)",
            input_names=["x"],
            dim=1,
        )

        # Maximize knapsack value.
        self.minimize_objective = False

        # Knapsack weight limit.
        if capacity_weight is not None:
            ineq_weight = MDOFunction(
                self._compute_weight_constraint,
                name="weight_surpass",
                f_type=MDOFunction.ConstraintType.INEQ,
                expr="sum(weights * x) - capacity_weight",
                input_names=["x"],
                dim=1,
            )
            self.add_ineq_constraint(ineq_weight)

        # Knapsack number of items limit.
        if capacity_items is not None:
            ineq_items = MDOFunction(
                self._compute_items_constraint,
                name="items_surpass",
                f_type=MDOFunction.ConstraintType.INEQ,
                expr="sum(x) - capacity_items",
                input_names=["x"],
                dim=1,
            )
            self.add_ineq_constraint(ineq_items)

    def _compute_weight_constraint(self, design_variables: ndarray) -> ndarray:
        """Compute the weight capacity constraint.

        Args:
            design_variables: The design variables vector.

        Returns:
            The knapsack weight surpass.
        """
        return atleast_1d(
            self.compute_knapsack_weight(design_variables) - self.capacity_weight
        )

    def _compute_items_constraint(self, design_variables: ndarray) -> ndarray:
        """Compute the number of items capacity constraint.

        Args:
            design_variables: The design variables vector.

        Returns:
            The knapsack number of items surpass.
        """
        return (
            atleast_1d(self.compute_knapsack_items(design_variables))
            - self.capacity_items
        )

    def compute_knapsack_value(self, design_variables: ndarray) -> ndarray:
        """Compute the knapsack total value.

        Args:
            design_variables: The design variables vector.

        Returns:
            The knapsack total value.
        """
        return atleast_1d(np_sum(self.values * design_variables))

    def compute_knapsack_weight(self, design_variables: ndarray) -> ndarray:
        """Compute the knapsack total weight.

        Args:
            design_variables: The design variables vector.

        Returns:
            The knapsack total weight.
        """
        return np_sum(self.weights * design_variables)

    @staticmethod
    def compute_knapsack_items(design_variables: ndarray) -> ndarray:
        """Compute the knapsack number of items.

        Args:
            design_variables: The design variables vector.

        Returns:
            The knapsack total number of items.
        """
        return np_sum(design_variables)


class MultiObjectiveKnapsack(Knapsack):
    """Multi-objective Knapsack optimization problem.

    With respect to the single-objective :class:`.Knapsack`, it adds an objective
    relative to the number of items packed. Therefore, besides maximizing the total
    knapsack value, one must also minimize the total number of items.

    All the variations of the :class:`.Knapsack` problem can still be achieved.
    """

    def __init__(
        self,
        values: ndarray,
        weights: ndarray,
        items_ub: ndarray | None = None,
        binary: bool = True,
        capacity_weight: float | None = None,
        capacity_items: int | None = None,
        initial_guess: ndarray | None = None,
    ) -> None:
        """The constructor.

        Initialize the MultiObjectiveKnapsack
        :class:`~gemseo.algos.opt_problem.OptimizationProblem` by defining the
        :class:`~gemseo.algos.design_space.DesignSpace` and the objective and
        constraint functions.

        The number of items in the problem is deduced from the :attr:`.values` array.

        Args:
            values: The items' values.
            weights: The items' weights.
            items_ub: The items' upper bounds.
                If None, an unlimited number of each item is allowed.
            binary: If True, the upper bound of design variables is set to 1.
            capacity_weight: The knapsack weight capacity.
                If None, the knapsack will have an unlimited weight capacity.
            capacity_items: The knapsack number of items capacity.
                If None, the knapsack will accept an unlimited total number of items.
            initial_guess: The initial guess for the optimal solution.
                If None, the initial guess will be an empty knapsack (0, 0, ..., 0).
        """
        super().__init__(
            values,
            weights,
            items_ub,
            binary,
            capacity_weight,
            capacity_items,
            initial_guess,
        )

        # Reset minimization goal.
        self.minimize_objective = True

        # Set objective function.
        self.objective = MDOFunction(
            self._compute_objective,
            name="knapsack",
            f_type=MDOFunction.FunctionType.OBJ,
            expr="[-sum(values * x), sum(x)]",
            input_names=["x"],
            dim=2,
        )

    def _compute_objective(self, design_variables: ndarray) -> ndarray:
        """Compute the objectives of the multi-objective Knapsack problem.

        - Maximize the knapsack total value.
        - Minimize the knapsack number of items.

        Args:
            design_variables: The design variables vector.

        Returns:
            The objective functions value.
        """
        obj = zeros(2)
        obj[0] = -self.compute_knapsack_value(design_variables)
        obj[1] = self.compute_knapsack_items(design_variables)
        return obj


def create_random_knapsack_problem(
    n_items: int,
    capacity_level: float = 0.1,
    binary: bool = True,
    obj_variant: str = "single",
) -> Knapsack | MultiObjectiveKnapsack:
    """Create a random :class:`.Knapsack` problem.

    One can also create a :class:`.MultiObjectiveKnapsack` problem by providing
    :attr:`.obj_variant` = 'multi'.

    The value and the weight of the items are integers randomly generated
    between 1 and 100.

    Args:
        n_items: The size of the set of items.
        capacity_level: The percentage of the set of items total weight
            corresponding to the knapsack capacity.
        binary: If True, only one unit of each item is allowed.
        obj_variant: Single-objective ('single') or multi-objective ('multi') problem.

    Returns:
        An instance of :class:`.Knapsack` or :class:`.MultiObjectiveKnapsack` depending
            on the :attr:`.obj_variant` provided.

    Raises:
        ValueError: Either if the number of items is not a positive integer or if the
            capacity_level is outside the range (0, 1).
    """
    if n_items < 1:
        raise ValueError("Number of items must be a positive number!")

    if not 0.0 < capacity_level < 1.0:
        raise ValueError("capacity_level must be in the interval (0, 1)!")

    rng = default_rng(1)
    values = rng.integers(1, 100, size=n_items)
    weights = rng.integers(1, 100, size=n_items)

    capacity_weight = capacity_level * sum(weights)

    if obj_variant == "multi":
        return MultiObjectiveKnapsack(values, weights, None, binary, capacity_weight)

    return Knapsack(values, weights, None, binary, capacity_weight)
