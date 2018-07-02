import time, itertools, math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from ema_workbench.em_framework.evaluators import LHS, SOBOL, MORRIS
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from SALib.analyze import sobol

def plot_input_outcome(inputs, outcomes, k, violin = True, q = 0.9):
    """this function plots:
    how a set of uncertainties have impact on the outcomes

    inputs: pd.DataFrame object

    outcomes: pd.DataFrame object

    k: index of outcomes' column

    violin: whether to plot in violin if the input is categorical

    q: threshold of being (un)desirable
    """
    plt.figure(figsize=(12,8))
    for i,col in enumerate(inputs.columns[:]):
        plt.subplot(3,6,i+1)
        x = inputs[col].values
        y = outcomes.iloc[:, k]

        # Violin plot for Categorical-value inputs
        if (x.dtype == "O"):
            if violin == True:
                sns.violinplot(x,y, color = "grey")
            else:
                plt.scatter(x, y, color = "grey", alpha = 0.5)

                # larger than 90% quantiles: Bad results
                y_q = y[y > y.quantile(q)]
                x_q = x[y_q.index]
                plt.scatter(x_q, y_q, color = "red", alpha = 0.5, edgecolor = "black")

        # Scatter plot for Real-value inputs
        else:
            plt.scatter(x, y, color = "grey", alpha = 0.5)

            # larger than 90% quantiles: Bad results
            y_q = y[y > y.quantile(q)]
            x_q = x[y_q.index]
            plt.scatter(x_q, y_q, color = "red", alpha = 0.5, edgecolor = "black")

        plt.xlabel(col)
        plt.ylabel("")
        plt.ylim(min(y), max(y))
        plt.grid()
    plt.suptitle(outcomes.columns[k], y=1.05, fontsize = 15)
    plt.tight_layout()

def plot_hist(outcomes, thresh = 0.9, abs_scale = False, ncols=2, nrows=5):
    ymax=int(outcomes.shape[0]*0.6)
    plt.figure(figsize = (ncols*5, nrows*5))
    for i,col in enumerate(outcomes):
        data = outcomes[col]
        plt.subplot(nrows,ncols,i+1)
        plt.hist(data, bins = 40)
        q = data.quantile(thresh)
        plt.vlines(x=q, ymin = 0, ymax=ymax, color = "red", linestyle="--")
        plt.ylim(0,ymax)
        if abs_scale:
            if "Damage" in data.name:
                plt.xlim(0,1e9)
            else:
                plt.xlim(0,1.0)
        plt.title(col)
        plt.grid()
    plt.tight_layout()

def plot_cont(inputs, outputs, k=0):
    df_outcome = outputs.copy()
    inputs = inputs[inputs.columns[inputs.dtypes != object]]
    df_input = inputs.copy()
    input_ = df_input.columns[k]

    ncols = int(df_outcome.shape[1] / 2)
    nrows = 2

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, sharex=True,
                             figsize=(ncols*4, nrows*4))
    locs = list(itertools.product(range(nrows), range(ncols)))
    i = 0
    for j, outcome_ in enumerate(df_outcome):
        if j == 5:
            i = i+1
        j = j%5
        loc = (i,j)
        ax = axes[loc]
        #specify x and y
        x = df_input[input_]
        y = df_outcome[outcome_]
        #scatter
        ax.scatter(x, y, s=5, alpha=0.4)

        #regression fit (1st order)
        fit = np.polyfit(x, y, deg=1)
        f = lambda x: fit[0]*x + fit[1]
        ax.scatter(x, f(x),s=1, alpha=0.2)

        #regression fit (2nd order)
        fit = np.polyfit(x, y, deg=2)
        f = lambda x: fit[0]*(x**2) + fit[1]*x + fit[2]
        ax.scatter(x, f(x),s=1, alpha=0.2)

        #details for being pretty!
        ax.set_title(y.name, fontsize=13)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.grid()
    fig.tight_layout()
    fig.suptitle("Regression Fit of {}(x) vs. Outcomes(y)".format(x.name), y=1.05, fontsize=25)

def plot_cat(inputs, outputs,k=0):
    cat_val = inputs.columns[inputs.dtypes == object]
    n_cat_val = len(cat_val)
    cat = cat_val[k]

    ncols = int(outputs.shape[1] / 2)
    nrows = 2

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, sharex=True,
                             figsize=(ncols*4, nrows*4))
    locs = list(itertools.product(range(nrows), range(ncols)))
    row = 0
    data = outputs.join(inputs[cat_val])
    for j, output in enumerate(outputs):
        if j == ncols:
            row += 1
        j = j%5
        loc = (row,j)
        ax = axes[loc]
        x = cat
        y = output
        sns.boxplot(x, y, data=data, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(output, fontsize=13)
        ax.grid()
    fig.tight_layout()
    fig.suptitle("Plot of {}(x) vs. Outcomes(y)".format(cat), y=1.05, fontsize=25)

def plot_sobol(model, sa_results, prob):
    if prob == "uncertainties":
        problem = get_SALib_problem(model.uncertainties)
    elif prob == "levers":
        problem = get_SALib_problem(model.levers)
    outcomes = sa_results[1]
    n_outcome = len(outcomes.keys())
    ncols=4
    nrows=math.ceil(n_outcome/ncols)

    fig, axes = plt.subplots(figsize=(ncols*4,nrows*4),
                             ncols=ncols, nrows=nrows)
    locs = list(itertools.product(range(nrows),range(ncols)))

    for i, key in enumerate(outcomes.keys()):
        loc = locs[i]
        ax = axes[loc]
        outcome = outcomes[key]
        Si = sobol.analyze(problem, outcome, calc_second_order=True, print_to_console=False)

        Si_filter = {k:Si[k] for k in ['ST','ST_conf','S1','S1_conf']}
        Si_df = pd.DataFrame(Si_filter, index=problem['names'])
        Si_df = Si_df.T.stack().reset_index()
        Si_df.columns = ["S", "Uncertainty", "Sensitivity Value"]
        Si_df["Uncertainty"] = Si_df["Uncertainty"].apply(lambda x: x[:9]).values
        sns.barplot(x = "Uncertainty", y = "Sensitivity Value",
                    data = Si_df, hue="S", ax=ax)
        ax.set_xlabel("")
        ax.set_xticklabels(labels=Si_df["Uncertainty"].unique(),rotation=90)
        ax.set_title(key)
        ax.grid()
    fig.tight_layout()
