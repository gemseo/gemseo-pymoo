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
"""Termination criteria for pymoo algorithms."""
from __future__ import annotations

from gemseo.algos.stop_criteria import TerminationCriterion


class DesignSpaceExploredException(TerminationCriterion):
    """Exception raised when the design space has been completely explored."""


class MaxGenerationsReached(TerminationCriterion):
    """Exception raised when the maximum number of generations is reached."""


class HyperVolumeToleranceReached(TerminationCriterion):
    """Exception raised when the hv_tol_rel or hv_tol_abs criteria is reached."""
