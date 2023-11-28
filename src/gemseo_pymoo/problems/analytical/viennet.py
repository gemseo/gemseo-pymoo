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
r"""**Viennet multi-objective problem**.

This module implements the Viennet multi-objective unconstrained problem:

.. math::

   \begin{aligned}
   \text{minimize the objective function }
   & f_1(x, y) = (x^2 + y^2) / 2 + sin(x^2 + y^2) \\
   & f_2(x, y) = (3x - 2y + 4)^2 / 8 + (x - y + 1)^2 / 27 + 15 \\
   & f_3(x, y) = 1 / (x^2 + y^2 + 1) - 1.1 e^{-(x^2 + y^2)} \\
   \text{with respect to the design variables }&x,\,y \\
   \text{subject to the bound constraints }
   & -3.0 \leq x \leq 3.0\\
   & -3.0 \leq y \leq 3.0
   \end{aligned}
"""

from __future__ import annotations

import logging

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from numpy import cos as np_cos
from numpy import exp as np_exp
from numpy import ndarray
from numpy import sin as np_sin
from numpy import zeros

LOGGER = logging.getLogger(__name__)


class Viennet(OptimizationProblem):
    """Viennet optimization problem."""

    def __init__(
        self, l_b: float = -3.0, u_b: float = 3.0, initial_guess: ndarray | None = None
    ) -> None:
        """The constructor.

        Initialize the Viennet :class:`~gemseo.algos.opt_problem.OptimizationProblem`
        by defining the :class:`~gemseo.algos.design_space.DesignSpace` and the
        objective function.

        Args:
            l_b: The lower bound (common value to all variables).
            u_b: The upper bound (common value to all variables).
            initial_guess: The initial guess for the optimal solution.
                If None, the initial guess will be (0., 0.).
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
            name="viennet",
            f_type=MDOFunction.FunctionType.OBJ,
            jac=self.compute_objective_jacobian,
            expr="[(x**2 + y**2) / 2 + sin(x**2 + y**2), 9*x - (y-1)**2,"
            "(3*x - 2*y + 4)**2 / 8 + (x - y + 1)^2 / 27 + 15,"
            "1 / (x**2 + y**2 + 1) - 1.1*exp(-(x**2 + y**2))]",
            input_names=["x", "y"],
            dim=3,
        )

    @staticmethod
    def compute_objective(design_variables: ndarray) -> ndarray:
        """Compute the objectives of the Viennet function.

        Args:
            design_variables: The design variables vector.

        Returns:
            The objective function value.
        """
        x, y = design_variables
        xy2 = x**2 + y**2

        obj = zeros(3)
        obj[0] = 0.5 * xy2 + np_sin(xy2)
        obj[1] = (3.0 * x - 2 * y + 4.0) ** 2 / 8.0 + (x - y + 1.0) ** 2 / 27.0 + 15.0
        obj[2] = 1.0 / (xy2 + 1.0) - 1.1 * np_exp(-xy2)
        return obj

    @staticmethod
    def compute_objective_jacobian(design_variables: ndarray) -> ndarray:
        """Compute the gradient of objective function.

        Args:
            design_variables: The design variables vector.

        Returns:
            The gradient of the objective functions wrt the design variables.
        """
        x, y = design_variables
        xy2 = x**2 + y**2

        jac = zeros([3, 2])
        jac[0, 0] = x + 2.0 * x * np_cos(xy2)
        jac[0, 1] = y + 2.0 * y * np_cos(xy2)
        jac[1, 0] = 3.0 * (3.0 * x - 2 * y + 4.0) / 4.0 + 2 * (x - y + 1.0) / 27.0
        jac[1, 1] = -2.0 * (3.0 * x - 2 * y + 4.0) / 4.0 - 2 * (x - y + 1.0) / 27.0
        jac[2, 0] = -2.0 * x * (xy2 + 1.0) ** (-2) + 1.1 * 2 * x * np_exp(-xy2)
        jac[2, 1] = -2.0 * y * (xy2 + 1.0) ** (-2) + 1.1 * 2 * y * np_exp(-xy2)
        return jac
