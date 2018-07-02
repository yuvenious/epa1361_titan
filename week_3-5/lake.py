import numpy as np
import math
from scipy.optimize import brentq

def lake_problem(
    b=0.42,          # decay rate for P in lake (0.42 = irreversible)
    q=2.0,           # recycling exponent
    mean=0.02,       # mean of natural inflows
    stdev=0.001,     # future utility discount rate
    delta=0.98,      # standard deviation of natural inflows
    alpha=0.4,       # utility from pollution
    nsamples=100,    # Monte Carlo sampling of natural inflows
        **kwargs):
    try:
        decisions = [kwargs[str(i)] for i in range(100)]
    except KeyError:
        decisions = [0, ] * 100

    Pcrit = brentq(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = len(decisions)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(decisions)
    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0

        natural_inflows = np.random.lognormal(
            math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
            math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
            size=nvars)

        for t in range(1, nvars):
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + \
                decisions[t-1] + natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)

        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)

    max_P = np.max(average_daily_P)
    utility = np.sum(alpha*decisions*np.power(delta, np.arange(nvars)))
    inertia = np.sum(np.absolute(np.diff(decisions)) < 0.02)/float(nvars-1)

    return max_P, utility, inertia, reliability

def process_p(values):
    values = np.asarray(values)
    values = np.mean(values, axis=0)
    return np.max(values)

def PredPrey(prey_birth_rate=0.025, predation_rate=0.0015, predator_efficiency=0.002,
             predator_loss_rate=0.06, initial_prey=50, initial_predators=20, dt=0.25, final_time=365, reps=1):

    #Initial values
    predators, prey, sim_time = [np.zeros((reps, int(final_time/dt)+1)) for _ in range(3)]
    
    for r in range(reps):
        predators[r,0] = initial_predators
        prey[r,0] = initial_prey

        #Calculate the time series
        for t in range(0, sim_time.shape[1]-1):

            dx = (prey_birth_rate*prey[r,t]) - (predation_rate*prey[r,t]*predators[r,t])
            dy = (predator_efficiency*predators[r,t]*prey[r,t]) - (predator_loss_rate*predators[r,t])

            prey[r,t+1] = max(prey[r,t] + dx*dt, 0)
            predators[r,t+1] = max(predators[r,t] + dy*dt, 0)
            sim_time[r,t+1] = (t+1)*dt
    
    #Return outcomes
    return {'TIME':sim_time,
            'predators':predators,
            'prey':prey}
