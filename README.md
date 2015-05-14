# econtools
*econtools* is a collection of econometric functions and convenient shortcuts
for data work with [pandas](http://github.com/pydata/pandas) and
[numpy](https://github.com/numpy/numpy). No warranty, express or implied, etc
etc.

## Econometrics
- OLS, 2SLS, LIML
- Option to absorb any variable via within-transformation (a la `areg` in Stata)
- Robust standard errors
  - HAC (`robust`/`hc1`, `hc2`, `hc3`)
  - Clustered standard errors (only one-way for now)
  - Spatial HAC (SHAC, aka Conley standard errors) with uniform and triangle
    kernels
- F-tests by variable name or `R` matrix.
- `binscatter`


## Other functions

- `econtools.merge` wraps `pandas.merge` and adds a variable that indicates for
  whether or not each row was matched in the merge.
  It takes an optional `assertval` argument that raises an error if, e.g., not
  all rows in the merge get matched.
- `read` and `write` use the passed file paths's extension to determine which
  `pandas` I/O method to use.

```python
import econtools
# Using econtools.read, these lines are equivalent
somedf = econtools.read('somefile.p')
somedf = pandas.read_pickle('somefile.p')
# These are also equivalent
newdf = econtools.read('newfile.csv')
newdf = pandas.read_csv('newfile.csv')
# This actually runs
for path in ['file1.csv', 'file2.dta', 'file3.p']:
    df = econtools.read(path)
    print df.describe()
```

