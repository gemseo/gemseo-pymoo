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
r"""**Chankong and Haimes multi-objective problem**.

This module implements the Chankong and Haimes multi-objective problem:

.. math::

   \begin{aligned}
   \text{minimize the objective function }
   & f_1(x, y) = 2 + (x - 2)^2 + (y - 1)^2 \\
   & f_2(x, y) = 9x - (y - 1)^2 \\
   \text{with respect to the design variables }&x,\,y \\
   \text{subject to the general constraints }
   & g_1(x, y) = x^2 + y^2 \leq 225.0\\
   & g_2(x, y) = x - 3y + 10 \leq 0.0\\
   \text{subject to the bound constraints }
   & -20.0 \leq x \leq 20.\\
   & -20.0 \leq y \leq 20.
   \end{aligned}

Chankong, V., & Haimes, Y. Y. (2008).
Multiobjective decision making: theory and methodology.
Courier Dover Publications.
"""

from __future__ import annotations

import logging

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from numpy import array
from numpy import ndarray
from numpy import zeros

LOGGER = logging.getLogger(__name__)


class ChankongHaimes(OptimizationProblem):
    """Chankong and Haimes optimization problem."""

    def __init__(
        self,
        l_b: float = -20.0,
        u_b: float = 20.0,
        initial_guess: ndarray | None = None,
    ) -> None:
        """The constructor.

        Initialize the ChankongHaimes
        :class:`~gemseo.algos.opt_problem.OptimizationProblem` by defining the
        :class:`~gemseo.algos.design_space.DesignSpace` and the objective and
        constraints functions.

        Args:
            l_b: The lower bound (common value to all variables).
            u_b: The upper bound (common value to all variables).
            initial_guess: The initial guess for the optimal solution.
        """
        design_space = DesignSpace()
        design_space.add_variable("x", size=1, l_b=l_b, u_b=u_b)
        design_space.add_variable("y", size=1, l_b=l_b, u_b=u_b)

        if initial_guess is None:
            design_space.set_current_value(zeros(2))
        else:
            design_space.set_current_value(initial_guess)

        super().__init__(design_space)

        # Set objective function.
        self.objective = MDOFunction(
            self.compute_objective,
            name="changkong_haimes",
            f_type=MDOFunction.FunctionType.OBJ,
            jac=self.compute_objective_jacobian,
            expr="[2 + (x-2)**2 + (y-1)**2, 9*x - (y-1)**2]",
            input_names=["x", "y"],
            dim=2,
        )

        ineq1 = MDOFunction(
            self.compute_constraint_1,
            name="ineq1",
            f_type=MDOFunction.ConstraintType.INEQ,
            jac=self.compute_constraint_1_jacobian,
            expr="x**2 + y**2 - 225",
            input_names=["x", "y"],
            dim=1,
        )
        self.add_ineq_constraint(ineq1)

        ineq2 = MDOFunction(
            self.compute_constraint_2,
            name="ineq2",
            f_type=MDOFunction.ConstraintType.INEQ,
            jac=self.compute_constraint_2_jacobian,
            expr="x - 3*y + 10",
            input_names=["x", "y"],
            dim=1,
        )
        self.add_ineq_constraint(ineq2)

    @staticmethod
    def compute_objective(design_variables: ndarray) -> ndarray:
        """Compute the objectives of the Chankong and Haimes function.

        Args:
            design_variables: The design variables vector.

        Returns:
            The objective function value.
        """
        obj = zeros(2)
        obj[0] = (
            2.0 + (design_variables[0] - 2.0) ** 2 + (design_variables[1] - 1.0) ** 2
        )
        obj[1] = 9.0 * design_variables[0] - (design_variables[1] - 1.0) ** 2
        return obj

    @staticmethod
    def compute_constraint_1(design_variables: ndarray) -> ndarray:
        """Compute the first constraint function.

        Args:
            design_variables: The design variables vector.

        Returns:
            The first constraint's value.
        """
        return array([design_variables[0] ** 2 + design_variables[1] ** 2 - 225.0])

    @staticmethod
    def compute_constraint_2(design_variables: ndarray) -> ndarray:
        """Compute the second constraint function.

        Args:
            design_variables: The design variables vector.

        Returns:
            The second constraint's value.
        """
        return array([design_variables[0] - 3 * design_variables[1] + 10.0])

    @staticmethod
    def compute_objective_jacobian(design_variables: ndarray) -> ndarray:
        """Compute the gradient of objective function.

        Args:
            design_variables: The design variables vector.

        Returns:
            The gradient of the objective functions wrt the design variables.
        """
        jac = zeros([2, 2])
        jac[0, 0] = 2.0 * (design_variables[0] - 2.0)
        jac[0, 1] = 2.0 * (design_variables[1] - 1.0)
        jac[1, 0] = 9.0
        jac[1, 1] = -2.0 * (design_variables[1] - 1.0)
        return jac

    @staticmethod
    def compute_constraint_1_jacobian(design_variables: ndarray) -> ndarray:
        """Compute the first inequality constraint jacobian.

        Args:
            design_variables: The design variables vector.

        Returns:
            The gradient of the first constraint function wrt the design variables.
        """
        jac = zeros([1, 2])
        jac[0, 0] = 2 * design_variables[0]
        jac[0, 1] = 2 * design_variables[1]
        return jac

    @staticmethod
    def compute_constraint_2_jacobian(design_variables: ndarray) -> ndarray:
        """Compute the second inequality constraint jacobian.

        Args:
            design_variables: The design variables vector.

        Returns:
            The gradient of the second constraint function wrt the design variables.
        """
        jac = zeros([1, 2])
        jac[0, 0] = 1.0
        jac[0, 1] = -3.0
        return jac
