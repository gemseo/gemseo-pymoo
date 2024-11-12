<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Changelog titles are:
- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.
-->

# Changelog

All notable changes of this project will be documented here.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0)
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# Version 4.0.0 (November 2024)

## Added

- Support GEMSEO v6.
- Support for Python 3.12.

# Removed

- The function `get_gemseo_opt_problem` is no longer available.

# Version 3.0.0 (March 2024)

# Changed

- The ``MultiObjectiveOptimizationResult`` class was moved to ``gemseo.algos.opt_result_multiobj``.
- The ``MultiObjectiveOptimizationResult.pareto`` was renamed to ``MultiObjectiveOptimizationResult.pareto_front``.
- The ``Pareto`` class was moved to ``gemseo.algos.pareto`` and renamed ``ParetoFront``.
- ``ParetoFront`` is now a dataclass.
- ``ParetoFront.problem`` is now a protected field.
- ``ParetoFront.front`` was renamed to ``ParetoFront.f_optima``.
- ``ParetoFront.set`` was renamed to ``ParetoFront.x_optima``.
- ``ParetoFront.utopia`` was renamed to ``ParetoFront.f_utopia``.
- ``ParetoFront.anti_utopia`` was renamed to ``ParetoFront.f_anti_utopia``.
- ``ParetoFront.anchor_front`` was renamed to ``ParetoFront.f_anchors``.
- ``ParetoFront.anchor_set`` was renamed to ``ParetoFront.x_anchors``.
- ``ParetoFront.min_norm_f`` was renamed to ``ParetoFront.f_utopia_neighbors``.
- ``ParetoFront.min_norm_x`` was renamed to ``ParetoFront.x_utopia_neighbors``.
- ``ParetoFront.min_norm`` was renamed to ``ParetoFront.distance_from_utopia``.
- ``ParetoFront.df_interest`` was renamed and is now a protected field ``ParetoFront._anchors_neighbors``.

# Version 2.0.0 (January 2024)

## Added

- Support for Python 3.10 and 3.11.
- Support for pymoo 0.6.1.

## Changed

- The ``diagram`` option of the ``MultiObjectiveDiagram`` is now called
  ``visualization`` to match Pymoo's naming.
- The ``diagram`` option of the ``MultiObjectiveDiagram`` is no longer a
  string, it must be a Pymoo ``Plot`` instance. This change is due to the
  removal of the visualization factory of the Pymoo API.
- The ``scalar_name`` option of the ``MultiObjectiveDiagram`` is now called
  ``decomposition`` to match Pymoo's naming.
- The ``scalar_name`` option of the ``MultiObjectiveDiagram`` is no longer a
  string, it must be a Pymoo ``Decomposition`` instance. This change is due to
  the removal of the visualization factory of the Pymoo API.
- A ``TypeError`` is now raised when the scalarization function provided with
  the ``decomposition`` option is not an instance of Pymoo's ``Decomposition``.
  In previous versions, a ``ValueError`` was raised.
- ``PymooOpt.EVOLUTIONARY_OPERATORS`` is now a dictionary with strings as keys
  and Pymoo ``Operator`` classes as values.
- Pymoo `0.6.0` removed support for evolutionary operator factories.
  Therefore, the options ``sampling``, ``selection``, ``mutation`` and
  ``crossover`` must now be instantiated by the user and then passed to the
  driver when the problem is executed.

## Removed

- Support for Python 3.8.
- Support for pymoo <0.6.1.
- The ``SCALARIZATION_FUNCTIONS`` class variable was removed from
  ``MultiObjectiveDiagram``.
- The ``EvolutionaryOperatorSimpleOptionsType`` was removed.
- The ``EvolutionaryOperatorOptionsType`` was removed.
- The ``EvolutionaryMixedOperatorTypes`` was removed.

## Fixed

- A typo in the ``Knapsack`` docstring was corrected.

# Version 1.1.3 (October 2023)

Update to GEMSEO 5.1.

# Version 1.1.2 (June 2023)

Update to GEMSEO 5.

# Version 1.1.1 (February 2023)

## Fixed

- The missing json file to validate the post processor options.
- The link to the Genetic Algorithm documentation.

# Version 1.1.0 (October 2022)

## Fixed

- The termination criterion `HyperVolumeToleranceReached` is now also
    taken into account for single-objective problems.

# Version 1.0.0 (July 2022)

First release.
