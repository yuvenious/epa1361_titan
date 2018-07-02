# import time, itertools
import re, functools
import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join
from ema_workbench import ScalarOutcome

def func(x):
    digit = re.sub(r".*_|\..*", "", x)
    if len(digit) == 1:
        digit = "".join(["0",digit])
        x = re.sub(r"\d", digit, x)
    return x

def varying_thresholds(th_death=1, th_cost="mean"):
    vfunc = np.vectorize(func)

    mypath = "./archive/worst"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = vfunc(onlyfiles)
    onlyfiles.sort()

    worst_results = pd.read_csv("".join([mypath, "/", onlyfiles[-1]]), index_col=0)
    deaths_thresh = dict(zip("Expected Number of Deaths(1/3)	Expected Number of Deaths(2/5)	Expected Number of Deaths(4)".split("	"),
                             [0.49659, 0.9921, 0.04]))
    outcomes = ['Expected Number of Deaths(1/3)', 'Expected Number of Deaths(2/5)', 'Expected Number of Deaths(4)',
                'Dike Investment Costs(1/3)', 'Dike Investment Costs(2/5)', 'Dike Investment Costs(4)',
                'RfR Total Costs', 'Expected Evacuation Costs']
    data = worst_results.loc[:, outcomes]
    locs = data.apply(lambda x: (x[0] < deaths_thresh["Expected Number of Deaths(1/3)"])
                      & (x[1] < deaths_thresh["Expected Number of Deaths(2/5)"])
                      & (x[2] < deaths_thresh["Expected Number of Deaths(4)"]), axis=1)

    deaths_thresh = pd.Series(deaths_thresh) * th_death
    if th_cost == "mean":
        thresholds = pd.Series(deaths_thresh).append(worst_results[locs].iloc[:, -5:].mean())
    else:
        thresholds = pd.Series(deaths_thresh).append(worst_results[locs].iloc[:, -5:].max() * th_cost)
    return thresholds

def robustness(threshold, data):
    return np.sum(data<=threshold)/data.shape[0]

def robustness_func_generator(thresholds):
    dict_funcs = {}
    for key, thresh in thresholds.items():
        dict_funcs[key] = functools.partial(robustness, thresh)

    direction = ScalarOutcome.MAXIMIZE
    robustness_functions = []
    outcomes = ['Expected Number of Deaths(1/3)', 'Expected Number of Deaths(2/5)', 'Expected Number of Deaths(4)',
                'Dike Investment Costs(1/3)', 'Dike Investment Costs(2/5)', 'Dike Investment Costs(4)',
                'RfR Total Costs', 'Expected Evacuation Costs']
    var_names = outcomes
    for i, var_name in enumerate(var_names):
        frac_name = "frac_{}".format(var_name)
        func = dict_funcs[var_name]

        robust_func = ScalarOutcome(
            name=frac_name,
            kind=direction,
            variable_name=var_name,
            function=func)

        robustness_functions.append(robust_func)
    return robustness_functions
