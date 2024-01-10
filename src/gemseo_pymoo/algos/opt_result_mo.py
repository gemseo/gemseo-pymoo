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
"""Multi-objective optimization result."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gemseo.algos.opt_result import OptimizationResult
from gemseo.algos.opt_result import Value
from gemseo.algos.pareto_front import compute_pareto_optimal_points
from gemseo.third_party.prettytable import PrettyTable
from gemseo.utils.string_tools import MultiLineString
from numpy import all as np_all
from numpy import argwhere
from numpy import array
from numpy import concatenate as np_concat
from numpy import max as np_max
from numpy import min as np_min
from numpy import ndarray
from numpy import zeros
from numpy.linalg import norm as np_norm
from pandas import DataFrame
from pandas import MultiIndex
from pandas import concat as pd_concat

if TYPE_CHECKING:
    from gemseo.algos.opt_problem import OptimizationProblem

LOGGER = logging.getLogger(__name__)


class Pareto:
    """Hold data from multi-objective optimization problems."""

    _df_interest: DataFrame
    """Hold the points of interest to be shown in the optimization result."""

    def __init__(
        self,
        problem: OptimizationProblem,
    ) -> None:
        """Initialize an object containing pareto related data.

        Args:
            problem: The optimization problem.
        """
        self._problem = problem

        # Pareto front and pareto set.
        self._front, self._set = self.get_pareto(problem)

        # Utopia and anti-utopia points.
        self._utopia = np_min(self._front, axis=0)
        self._anti_utopia = np_max(self._front, axis=0)

        # Anchor points (extremities of the Pareto front).
        self._anchor_front, self._anchor_set = self.get_pareto_anchor(
            self._front, self._set
        )

        # Point with the lowest norm (closest to the utopia point).
        self._min_norm_f, self._min_norm_x, self._min_norm = self.get_lowest_norm(
            self._front, self._set, self._utopia
        )

        # Get dataset as a dataframe.
        df = self._problem.to_dataset()
        df.index = df.index.astype(int)

        # Get design variables group,
        # and reorder the columns to match the design space order.
        df_dp = df.get_view(variable_names=problem.design_space.variable_names)
        ind_anchor = [df.index[np_all(df_dp == p, axis=1)][0] for p in self._anchor_set]
        ind_min_norm = [
            df.index[np_all(df_dp == p, axis=1)][0] for p in self._min_norm_x
        ]

        # DataFrame with points of interest.
        df_anchor = df.loc[ind_anchor].droplevel(0, axis=1)
        df_anchor.index = [f"anchor_{i + 1}" for i in range(len(ind_anchor))]
        df_min_norm = df.loc[ind_min_norm].droplevel(0, axis=1)
        df_min_norm.index = [f"compromise_{i + 1}" for i in range(len(ind_min_norm))]
        self._df_interest = pd_concat([df_anchor, df_min_norm], axis=0)

        # Shift dimensions to start at 1.
        new_columns = [
            (*c[0:-1], str(int(c[-1]) + 1)) for c in self._df_interest.columns
        ]
        self._df_interest.columns = MultiIndex.from_tuples(new_columns)

    @property
    def problem(self) -> OptimizationProblem:
        """The optimization problem whose Pareto data is represented."""
        return self._problem

    @property
    def front(self) -> ndarray:
        """The values of the objectives of all pareto efficient solutions."""
        return self._front

    @property
    def set(self) -> ndarray:  # noqa: A003
        """The values of the design variables of all pareto efficient solutions."""
        return self._set

    @property
    def anchor_front(self) -> ndarray:
        """The values of the objectives of all anchor points.

        At those points, each objective is minimized one at a time.
        """
        return self._anchor_front

    @property
    def anchor_set(self) -> ndarray:
        """The values of the design variables values of all anchor points.

        At those points, each objective is minimized one at a time.
        """
        return self._anchor_set

    @property
    def utopia(self) -> ndarray:
        """The ideal point where every objective reaches its minimum simultaneously."""
        return self._utopia

    @property
    def anti_utopia(self) -> ndarray:
        """The point where every objective reaches its maximum simultaneously."""
        return self._anti_utopia

    @property
    def min_norm_f(self) -> ndarray:
        """The objectives value of the closest point(s) to the :attr:`.utopia`."""
        return self._min_norm_f

    @property
    def min_norm_x(self) -> ndarray:
        """The design variables value of the closest point(s) to the :attr:`.utopia`."""
        return self._min_norm_x

    @property
    def min_norm(self) -> float:
        """The shortest distance (2-norm) from the pareto front to the utopia point."""
        return self._min_norm

    @staticmethod
    def get_lowest_norm(
        pareto_front: ndarray,
        pareto_set: ndarray,
        reference: ndarray | None = None,
        order: int = 2,
    ) -> tuple[ndarray, ndarray, float]:
        """Get Pareto points with the lowest norm relative to a reference point.

        Args:
            pareto_front: The objectives' value of all non-dominated points.
            pareto_set: The design variables' value of all non-dominated points.
            reference: The reference point.
                If None, the origin (0, 0, ..., 0) will be used.
            order: The order of the norm.

        Returns:
            The objectives' values of the point(s) with the lowest norm.
            The design variables' values of the point(s) with the lowest norm.
            The lowest norm value.

        Raises:
            ValueError: If the reference point does not have the appropriate dimension.
        """
        if reference is None:
            reference = zeros(pareto_front.shape[1])

        if reference.shape != (pareto_front.shape[1],):
            raise ValueError(
                f"Reference point {reference} does not have the "
                "same amount of objectives as the pareto front"
            )

        pareto_norm = np_norm(pareto_front - reference, axis=1, ord=order)
        min_pareto_norm = np_min(pareto_norm)

        # Get all indexes (if more than one).
        index_min = argwhere(pareto_norm == min_pareto_norm).flatten()

        return pareto_front[index_min], pareto_set[index_min], min_pareto_norm

    @staticmethod
    def get_pareto(
        gemseo_problem: OptimizationProblem,
    ) -> tuple[None, None] | tuple[ndarray, ndarray]:
        """Get Pareto Front and Pareto Set from the database.

        Args:
            gemseo_problem: The optimization problem containing the results
                from an optimization run.

        Returns:
            The objectives' value of all non-dominated points and
            the design variables' value of all non-dominated points.

            `None` if a single-objective
            :class:`~gemseo.algos.opt_problem.OptimizationProblem` is provided.

        Raises:
            RuntimeError: If the optimization problem is single-objective.
        """
        if gemseo_problem.objective.dim == 1:
            raise RuntimeError("Single-objective problems have no Pareto Front.")

        n_iter = len(gemseo_problem.database)
        constraints = (
            gemseo_problem.get_ineq_constraints() + gemseo_problem.get_eq_constraints()
        )

        dv_history = zeros((n_iter, gemseo_problem.design_space.dimension))
        obj_history = zeros((n_iter, gemseo_problem.objective.dim))
        feasibility = zeros(n_iter)

        iteration = 0
        for x_vect, out_val in gemseo_problem.database.items():
            dv_history[iteration] = x_vect.unwrap()
            if gemseo_problem.objective.name in out_val:
                obj_history[iteration] = array(out_val[gemseo_problem.objective.name])
                feasibility[iteration] = gemseo_problem.is_point_feasible(
                    out_val, constraints=constraints
                )
            else:
                obj_history[iteration] = float("nan")
                feasibility[iteration] = False
            iteration += 1

        pareto_opt_pts = compute_pareto_optimal_points(obj_history, feasibility)
        pareto_front = obj_history[pareto_opt_pts]
        pareto_set = dv_history[pareto_opt_pts]

        return pareto_front, pareto_set

    @staticmethod
    def get_pareto_anchor(
        pareto_front: ndarray,
        pareto_set: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Get Pareto's anchor points.

        Args:
            pareto_front: The objectives' value of all non-dominated points.
            pareto_set: The design variables' value of all non-dominated points.

        Returns:
            The objectives' values of all anchor points.
            The design variables' values of all anchor points.
        """
        n_obj = pareto_front.shape[1]

        anchor_points_index = zeros(n_obj, dtype=int)
        min_pf = np_min(pareto_front, axis=0)
        for obj_i in range(n_obj):
            anchor_points_index[obj_i] = argwhere(
                pareto_front[:, obj_i] == min_pf[obj_i]
            )[0]

        return pareto_front[anchor_points_index], pareto_set[anchor_points_index]

    @staticmethod
    def get_pretty_table_from_df(
        df: DataFrame,
    ) -> PrettyTable:
        """Build a tabular view of the Pareto problem.

        Args:
            df: The Pareto data.

        Returns:
            A :class:`~gemseo.third_party.prettytable.PrettyTable`
                representing the dataframe.
        """
        fields = [df.index.name or "name"]
        if df.columns.nlevels == 1:
            fields += list(df.columns)
        else:
            fields += [f"{col[0]} ({', '.join(col[1:])})" for col in df.columns]

        table = PrettyTable(fields)
        table.float_format = "%.6g"
        for _, row in df.iterrows():
            name = row.name
            if isinstance(name, tuple):
                content = [f"{name[0]} ({', '.join(name[1:])})"]
            else:
                content = [name]
            content += list(row.values)
            table.add_row(content)
        table.align = "r"
        return table

    def __str__(self) -> str:
        obj_names = [self._problem.get_objective_name()]
        c_names = self._problem.get_constraint_names()
        dv_names = self._problem.get_design_variable_names()

        msg = MultiLineString()
        msg.add(
            "Pareto optimal points : {} / {}",
            self._front.shape[0],
            len(self._problem.database),
        )
        msg.add("Utopia point : {}", self._utopia)
        msg.add("Compromise solution (closest to utopia) : {}", self._min_norm_f)
        msg.add("Compromise solution norm : {}", self._min_norm)
        msg.add("Objectives value:")
        msg.indent()
        for line in str(
            self.get_pretty_table_from_df(self._df_interest[obj_names].T)
        ).split("\n"):
            msg.add("{}", line)
        if self._problem.constraints:
            msg.dedent()
            msg.add("Constraint(s) value:")
            msg.indent()
            for line in str(
                self.get_pretty_table_from_df(self._df_interest[c_names].T)
            ).split("\n"):
                msg.add("{}", line)
        msg.dedent()
        msg.add("Design space:")
        msg.indent()

        # Prepare DataFrame for design space.
        ds = self._problem.design_space
        cols = MultiIndex.from_tuples([
            (n, str(d + 1)) for n in dv_names for d in range(ds.get_size(n))
        ])
        types = np_concat([ds.get_type(var) for var in dv_names])
        df_lb = DataFrame(
            ds.get_lower_bounds().reshape(1, -1), columns=cols, index=["lower_bound"]
        )
        df_ub = DataFrame(
            ds.get_upper_bounds().reshape(1, -1), columns=cols, index=["upper_bound"]
        )
        df_types = DataFrame(types.reshape(1, -1), columns=cols, index=["type"])
        df_interest_dv = pd_concat([
            df_lb,
            self._df_interest[dv_names],
            df_ub,
            df_types,
        ])

        for line in str(self.get_pretty_table_from_df(df_interest_dv.T)).split("\n"):
            msg.add("{}", line)
        return str(msg)


@dataclass
class MultiObjectiveOptimizationResult(OptimizationResult):
    """The result of a multi-objective optimization."""

    pareto: Pareto | None = None
    """The pareto efficient solutions."""

    def __repr__(self) -> str:
        msg = MultiLineString()
        msg.add("Multi-objective optimization result:")
        msg.indent()
        msg.add("Design variables: {}", self.x_opt)
        msg.add("Objective function: {}", self.f_opt)
        msg.add("Feasible solution: {}", self.is_feasible)
        msg.add("Pareto available: {}", self.pareto is not None)
        if self.pareto is not None:
            msg.indent()
            msg.add("Number of point: {}", self.pareto.front.shape[0])
            msg.add("Lowest norm: {}", self.f_opt)
            msg.add("Design variables: {}", self.x_opt)
            msg.dedent()
        return str(msg)

    def __str__(self) -> str:
        msg = MultiLineString()
        msg.add("Multi-objective optimization result:")
        msg.indent()
        msg.add("Optimizer info:")
        msg.indent()
        msg.add("Status: {}", self.status)
        msg.add("Message: {}", self.message)
        msg.add("The result is {}.", "feasible" if self.is_feasible else "unfeasible")
        msg.add(
            "Number of calls to the objective function by the optimizer: {}",
            self.n_obj_call,
        )
        msg.dedent()
        msg.add("Pareto efficient solutions:")
        msg.indent()
        for line in str(self.pareto).split("\n"):
            msg.add("{}", line)
        return str(msg)

    def to_dict(self) -> dict[str, Value]:
        """Convert the multi-objective optimization result to a dictionary.

        The keys are the names of the optimization result fields,
        except for the constraint values, gradients and the :attr:`.pareto`.
        The key ``"constr:y"`` maps to ``result.constraint_values["y"]``,
        ``"constr_grad:y"`` maps to ``result.constraints_grad["y"]`` and
        ``"pareto:y"`` maps to ``result.pareto.y``.

        Returns:
            A dictionary representation of the optimization result.
        """
        dict_ = super().to_dict()

        # Remove pareto attribute.
        pareto = dict_.pop("pareto")

        # Breakdown pareto attribute into its own attributes.
        if pareto is not None:
            for attr, value in self.pareto.__dict__.items():
                if isinstance(value, (int, float, str, bool, list, tuple, ndarray)):
                    dict_[f"pareto:{attr}"] = value

        return dict_

    get_data_dict_repr = to_dict
