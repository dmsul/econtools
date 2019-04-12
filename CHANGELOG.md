# Changelog

## [0.1.1] - 2019-04-12

### Added
- Regression `Results` objects now display formatted regression summary when
  printed directly (i.e. `__repr__` method was added).
- Raw lists of state FIPS, abbreviations, and names.

### Changed
- The `a_name` keyword argument in regression methods has been deprecated in
  favor of the new `fe_name` keyword argument.
- A lot of metrics methods were refactored. This should not impact end users
  but should make extending those methods a little easier.

## [0.1.0] - 2018-09-08

### Added
- Tests for `load_or_build`.

### Changed
- Tests now use `pytest` instead of `nose`.
- `outreg` now defaults to adding all regressors to the table instead of
  requiring a specific list.


## [0.0.1] - 2018-08-31
Initial public pseudo-release.
