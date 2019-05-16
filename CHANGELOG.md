# Changelog

## Next

### Added
- Mypy typing hints.
- New function `winsorize`.
- Missing state name/abbreviation/fips crosswalk functions.

### Changed
- REF: Removed Python 2 compatibility (following Pandas). `econtools` now
  requires Python >=3.6.
- ENH: `load_or_build` messages are now more informative.
- ENH: Updated docstrings for several functions.
- BUG: `state_name_to_fips` was not made accessible for external use.


## [0.1.1] - 2019-04-12

### Added
- Regression `Results` objects now display formatted regression summary when
  printed directly (i.e. `__repr__` method was added).
- Raw lists of state FIPS, abbreviations, and names.

### Changed
- REF: The `a_name` keyword argument in regression methods has been deprecated in
  favor of the new `fe_name` keyword argument.
- REF: A lot of metrics methods were refactored. This should not impact end users
  but should make extending those methods a little easier.

## [0.1.0] - 2018-09-08

### Added
- Tests for `load_or_build`.

### Changed
- REF: Tests now use `pytest` instead of `nose`.
- ENH: `outreg` now defaults to adding all regressors to the table instead of
  requiring a specific list.


## [0.0.1] - 2018-08-31
Initial public pseudo-release.
