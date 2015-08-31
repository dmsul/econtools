set more off
set trace off
clear all

prog def format_results_header
    args filename handle getconst
    if "`getconst'" == "" local getconst = 0
    else local getconst = 1
    local xs $X
    if `getconst' == 1 local xs $X _cons

    file open `handle' using `filename', w replace text
    file write `handle' "import pandas as pd" _n "import numpy as np" _n(2)
    file write `handle' "class regout(object):" _n(2)
    file write `handle' _tab "def __init__(self, **kwargs):" _n
    file write `handle' _tab(2) "self.__dict__.update(kwargs)"
    file write `handle' _n(4)

    * Write df index/cols once (easier to do quick manual edits)
    file write `handle' "stat_names=['coeff', 'se', 't', 'p>t', 'CI_low', 'CI_high']" _n
    file write `handle' "var_names=["
    foreach x in $X {
        file write `handle' "'`x'', "
    }
    if `getconst' == 1 file write `handle' "'_cons'"
    file write `handle' "]" _n
end

prog def format_results
    args handle name getconst
    if "`getconst'" == "" local getconst = 0
    else local getconst = 1
    local xs $X
    if `getconst' == 1 local xs $X _cons

    file write `handle' "`name' = regout(" _n

    * Write summary output table
    file write `handle'  "summary=pd.DataFrame(np.array([" _n
    mat t = r(table)
    foreach x in `xs' {
        file write `handle' "["
        foreach stat in b se t pvalue ll ul {
            mat temp = t["`stat'", "`x'"]
            local temp2 = temp[1, 1]
            file write `handle' "`temp2'," _n
        }
        file write `handle' "]," _n
    }
    file write `handle' "])," _n
    file write `handle' "columns=stat_names," _n
    file write `handle' "index=var_names)," _n

    * Write VCE matrix
    mat V = e(V)
    file write `handle'  "vce=pd.DataFrame(np.array([" _n
    foreach x1 in `xs' {
        file write `handle' "["
        foreach x2 in `xs' {
            mat temp = V["`x1'", "`x2'"]
            local temp2 = temp[1, 1]
            file write `handle' "`temp2'," _n
        }
        file write `handle' "]," _n
    }
    file write `handle' "])," _n
    file write `handle' "columns=var_names," _n
    file write `handle' "index=var_names)," _n

    * Write individual stats
    foreach stat in N r2 r2_a mss tss rss kappa {
        local temp2 = e(`stat')
        if "`temp2'" == "." local temp2 np.nan
        file write `handle'  "`stat'=`temp2'," _n
    }

    qui test $X
    foreach stat in F p {
        local temp2 = r(`stat')
        if "`stat'" == "p" local statname = "pF"
        else local statname = "`stat'"
        file write `handle'  "`statname'=`temp2'," _n
    }
    file write `handle' ")" _n
end

sysuse auto

global Y price
global X mpg length
global Z trunk weight
global cluster gear_ratio

* OLS
local reg_type ols
format_results_header "src_`reg_type'.py" `reg_type'_output 1

qui reg $Y $X,
format_results `reg_type'_output "`reg_type'_std" 1
qui reg $Y $X, robust
format_results `reg_type'_output "`reg_type'_robust" 1
qui reg $Y $X, vce(hc2)
format_results `reg_type'_output "`reg_type'_hc2" 1
qui reg $Y $X, vce(hc3)
format_results `reg_type'_output "`reg_type'_hc3" 1
qui reg $Y $X, cluster($cluster)
format_results `reg_type'_output "`reg_type'_cluster" 1

file close `reg_type'_output

* tsls
local reg_type tsls
format_results_header "src_`reg_type'.py" `reg_type'_output 1

qui ivreg $Y ($X = $Z),
format_results `reg_type'_output "`reg_type'_std" 1
qui ivreg $Y ($X = $Z), robust
format_results `reg_type'_output "`reg_type'_robust" 1
qui ivreg $Y ($X = $Z), cluster($cluster)
format_results `reg_type'_output "`reg_type'_cluster" 1

file close `reg_type'_output

* areg
local reg_type areg
format_results_header "src_`reg_type'.py" `reg_type'_output

bys $cluster: gen T = _N
gen dum = _n
xtset $cluster dum

qui xtivreg2 $Y $X, fe small
format_results `reg_type'_output "`reg_type'_std"
qui xtivreg2 $Y $X, fe small robust
format_results `reg_type'_output "`reg_type'_robust"
qui xtivreg2 $Y $X, fe small cluster($cluster)
format_results `reg_type'_output "`reg_type'_cluster"

file close `reg_type'_output

* xtivreg
// Stata's `xtivreg` get's DoF wrong;
// `xtivreg2` doesn't use r(table) if there are exluded instruments
local reg_type atsls
format_results_header "src_`reg_type'.py" `reg_type'_output

qui xtivreg $Y ($X = $Z) if T > 1, fe small
format_results `reg_type'_output "`reg_type'_std"
/*
// These are wrong, don't know why
qui xtivreg $Y ($X = $Z), fe vce(robust)
format_results `reg_type'_output "`reg_type'_robust"
qui xtivreg $Y ($X = $Z), fe cluster($cluster)
format_results `reg_type'_output "`reg_type'_cluster"
*/

file close `reg_type'_output

*** LIML ***
global Z trunk weight headroom
* all at once
local reg_type liml
format_results_header "src_`reg_type'.py" `reg_type'_output 1

qui ivregress liml $Y ($X = $Z), small
format_results `reg_type'_output "`reg_type'_std" 1
qui ivregress liml $Y ($X = $Z), robust small
format_results `reg_type'_output "`reg_type'_robust" 1
qui ivregress liml $Y ($X = $Z), cluster($cluster) small
format_results `reg_type'_output "`reg_type'_cluster" 1

file close `reg_type'_output
