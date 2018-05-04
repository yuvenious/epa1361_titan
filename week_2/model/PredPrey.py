"""
Python model "PredPrey.py"
Translated using PySD version 0.8.3
"""
from __future__ import division
import numpy as np
from pysd import utils
import xarray as xr

from pysd.py_backend.functions import cache
from pysd.py_backend import functions

_subscript_dict = {}

_namespace = {
    'TIME': 'time',
    'Time': 'time',
    'predator_growth': 'predator_growth',
    'predators': 'predators',
    'prey': 'prey',
    'prey_growth': 'prey_growth',
    'predator_loss': 'predator_loss',
    'predator_efficiency': 'predator_efficiency',
    'prey_loss': 'prey_loss',
    'initial_predators': 'initial_predators',
    'initial_prey': 'initial_prey',
    'predator_loss_rate': 'predator_loss_rate',
    'prey_birth_rate': 'prey_birth_rate',
    'predation_rate': 'predation_rate',
    'FINAL_TIME': 'final_time',
    'INITIAL_TIME': 'initial_time',
    'SAVEPER': 'saveper',
    'TIME_STEP': 'time_step'
}

__pysd_version__ = "0.8.3"


@cache('step')
def predator_growth():
    """
    predator_growth



    component


    """
    return predator_efficiency() * predators() * prey()


@cache('step')
def predators():
    """
    predators

    [0,?]

    component


    """
    return integ_predators()


@cache('step')
def prey():
    """
    prey

    [0,?]

    component


    """
    return integ_prey()


@cache('step')
def prey_growth():
    """
    prey_growth



    component


    """
    return prey_birth_rate() * prey()


@cache('step')
def predator_loss():
    """
    predator_loss



    component


    """
    return predator_loss_rate() * predators()


@cache('run')
def predator_efficiency():
    """
    predator_efficiency



    constant


    """
    return 0.002


@cache('step')
def prey_loss():
    """
    prey_loss



    component


    """
    return predation_rate() * predators() * prey()


@cache('run')
def initial_predators():
    """
    initial_predators



    constant


    """
    return 20


@cache('run')
def initial_prey():
    """
    initial_prey



    constant


    """
    return 50


@cache('run')
def predator_loss_rate():
    """
    predator_loss_rate



    constant


    """
    return 0.06


@cache('run')
def prey_birth_rate():
    """
    prey_birth_rate



    constant


    """
    return 0.025


@cache('run')
def predation_rate():
    """
    predation_rate



    constant


    """
    return 0.0015


@cache('run')
def final_time():
    """
    FINAL_TIME

    Day

    constant

    The final time for the simulation.
    """
    return 365


@cache('run')
def initial_time():
    """
    INITIAL_TIME

    Day

    constant

    The initial time for the simulation.
    """
    return 0


@cache('step')
def saveper():
    """
    SAVEPER

    Day [0,?]

    component

    The frequency with which output is stored.
    """
    return time_step()


@cache('run')
def time_step():
    """
    TIME_STEP

    Day [0,?]

    constant

    The time step for the simulation.
    """
    return 0.25


integ_predators = functions.Integ(lambda: predator_growth() - predator_loss(),
                                  lambda: initial_predators())

integ_prey = functions.Integ(lambda: prey_growth() - prey_loss(), lambda: initial_prey())
