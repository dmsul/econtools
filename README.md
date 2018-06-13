# econtools
*econtools* is a Python package of econometric functions and convenient
shortcuts for data work with [pandas](http://github.com/pydata/pandas) and
[numpy](https://github.com/numpy/numpy). Full documentation
[here](http://www.danielmsullivan.com/econtools).

## Econometrics
- OLS, 2SLS, LIML
- Option to absorb any variable via within-transformation (a la `areg` in Stata)
- Robust standard errors
  - HAC (`robust`/`hc1`, `hc2`, `hc3`)
  - Clustered standard errors
  - Spatial HAC (SHAC, aka Conley standard errors) with uniform and triangle
    kernels
- F-tests by variable name or `R` matrix.
- Local linear regression.

```python
import econtools
import econtools.metrics as mt

# Read Stata DTA file
df = econtools.read('my_data.dta')

# Estimate OLS regression with fixed-effects and clustered s.e.'s
result = mt.reg(df,                     # DataFrame to use
                'y',                    # Outcome
                ['x1', 'x2'],           # Indep. Variables
                a_name='person_id',     # Fixed-effects using variable 'person_id'
                cluster='state'         # Cluster by state
)

# Results
print(result.summary)                                # Print regression results
beta_x1 = result.beta['x1']                          # Get coefficient by variable name
r_squared = result.r2a                               # Get adjusted R-squared
joint_F = result.Ftest(['x1', 'x2'])                 # Test for joint significance
equality_F = result.Ftest(['x1', 'x2'], equal=True)  # Test for coeff. equality
```

## Regression and Summary Stat Tables

- `outreg` takes a regression results and creates a LaTeX-formatted tabular
  fragment.
- `table_statrow` can be used to add arbitrary statistics, notes, etc. to a
  table. Can also be used to create a table of summary statistics.
- `write_notes` makes it easy to save table notes that depend on your data.

```python
from econtools import read, outreg, table_statrow, write_notes
import econtools.metrics as mt

# Load data
df = read('my_data.dta')

# Run Regressions
reg1 = mt.reg(df, 'ln_wage', ['ed'], addcons=True)
reg2 = mt.reg(df, 'ln_wage', ['age', 'age2'], addcons=True)

# Put coefficients and standard errors in a table
regs = (reg1, reg2)
table_string = outreg(regs,
                      ['ed', 'age', 'age2', '_cons'],     # Add these coefficients to the table
                      ['Years Education',                 # Use these label for the coeffs
                       'Age',
                       'Age$^2$',
                       'Constant']
                      digits=3                            # Round to 3 decimal digits.
                      )

# Add R^2 to the table
table_string += "\\\\ \n"                           # Empty line between betas and r2
table_string += table_statrow("R^2",                # Add a row with this label
                              [x.r2 for x in regs], # Fill the row with these values
                              digits=3)

# Save the table string to a file
results_path = 'my_results.tex'
with open(results_path) as f:
    f.write(table_string)

# Save separate file with table notes
notes = "Sample size is {}.".format(reg1.N)
write_notes(notes, results_path)
```
The above snippet of code saves two files to disk, `my_results.tex`:
```latex
Years Education  & 3.142**    &             \\
                 & (1.413)    &             \\
Age              &            & 1.590       \\
                 &            & (2.020)     \\
Age$^2$          &            & 1.390       \\
                 &            & (4.024)     \\
Constant         & 11.132***  & 10.324***   \\
                 & (1.324)    & (2.720)     \\
\\
R^2              & 0.129      & 0.132       \\
```

and `my_results_notes.tex`:
```latex
Sample size is 934.
```
These files can be incorporated into a LaTeX document using the `include`
command.


## Data I/O

- `read` and `write`: Use the passed file path's extension to determine which
  `pandas` I/O method to use. Useful for writing functions that
  programmatically read DataFrames from disk which are saved in different
  formats. See examples above and below.
- `load_or_build`: A function decorator that caches datasets to disk.
  This function builds the requested dataset and saves it to disk if it
  doesn't already exist on disk. If the dataset is already saved, it simply
  loads it, saving computational time and allowing the use of a single function
  to both load and build data.
  ```python
  from econtools import load_or_build, read

  @load_or_build('my_data_file.dta')
  def build_my_data_file():
    """
    Cleans raw data from CSV format and saves as Stata DTA.
    """
    df = read('raw_data.csv')
    # Clean the DataFrame
    return df
  ```
  File type  is automatically detected from the passed filename. In this case,
  Stata DTA from `my_data_file.dta`.
- `save_cli`: Simple wrapper for `argparse` that let's you use a `--save` flag
  on the command line. This lets you run a regression without over-writing the
  previous results and without modifying the code in any way (i.e., commenting
  out the "save" lines).

  In your regression script:
  ```python
  from econtools import save_cli

  def regression_table(save=False):
    """ Run a regression and save output if `save == True`.  """ 
    # Regression guts


  if __name__ == '__main__':
      save = save_cli()
      regression_table(save=save)
  ```
  In the command line/bash script:
  ```bash
  python run_regression.py          # Runs regression without saving output
  python run_regression.py --save   # Runs regression and saves output
  ```

## Coming soon

- Simple Kriging
- Prettier regression output
