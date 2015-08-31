import pandas as pd
import numpy as np

class regout(object):

	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)



stat_names=['coeff', 'se', 't', 'p>t', 'CI_low', 'CI_high']
var_names=['mpg', 'length', ]
atsls_std = regout(
summary=pd.DataFrame(np.array([
[-49467.76193509437,
1937433.402520039,
-.0255326257257417,
.9797711358109091,
-3978764.822885125,
3879829.299014937,
],
[-8280.069027964491,
327594.5105698549,
-.0252753595094167,
.9799749166015559,
-672672.5306688137,
656112.3926128848,
],
]),
columns=stat_names,
index=var_names),
vce=pd.DataFrame(np.array([
[3753648189200.375,
634678341446.1863,
],
[634678341446.1863,
107318163355.5028,
],
]),
columns=var_names,
index=var_names),
N=51,
r2=np.nan,
r2_a=np.nan,
mss=np.nan,
tss=np.nan,
rss=973193578332.4143,
kappa=np.nan,
F=.0010619545509911,
pF=.9989386404150321,
)
