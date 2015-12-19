# econtools
*econtools* is a Python package of econometric functions and convenient
shortcuts for data work with [pandas](http://github.com/pydata/pandas) and
[numpy](https://github.com/numpy/numpy).

## Econometrics
- OLS, 2SLS, LIML
- Option to absorb any variable via within-transformation (a la `areg` in Stata)
- Robust standard errors
  - HAC (`robust`/`hc1`, `hc2`, `hc3`)
  - Clustered standard errors
  - Spatial HAC (SHAC, aka Conley standard errors) with uniform and triangle
    kernels
- F-tests by variable name or `R` matrix.


## I/O

- `load_or_build`: If a dataset exists at the passed path, load it, otherwise
  build it using the passed function. Allows to datasets to be built from
  scratch or accessed using a single function call.
- `save_cli`: Simple wrapper for `argparse` that let's you use a `--save` flag
  on the command line, (e.g., to save results of a regression script to disk).
- `read` and `write`: Use the passed file paths's extension to determine which
  `pandas` I/O method to use. Useful for writing functions that
  programmatically read DataFrames from disk which are saved in different
  formats.

## Working with DataFrames

- `econtools.merge`: Wraps `pandas.merge` and adds a variable that indicates
  whether or not each row was matched in the merge.
  It takes an optional `assertval` argument that raises an error if, e.g., not
  all rows in the merge get matched.
- `group_id`: generate a unique ID number from passed variables.
