from .bergsf import *
from .etete import a_fun
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from . import example

@dataclass
class GaussianComponent: 
    weight: float
    mean: np.ndarray
    cov: np.ndarray

def example_setup_gc():

    
    f = 2 * np.eye(1)
    q = 1 * np.eye(1)
    h = np.array([[1.]])
    r = np.eye(1)

    pb = 0.01
    ps = 0.99
    pd = 0.6
    
    birth_model = [
            GaussianComponent(1.0, np.zeros(1), np.eye(1)),
            GaussianComponent(1.0, 1 + np.zeros(1), np.eye(1)),
    ]
    clutter_mean = np.zeros(1)
    clutter_var = np.eye(1)
    llambda = 2
    
    b = BerGSFGaussianClutter(f, q, h, r, birth_model, llambda, clutter_mean, clutter_var, ps, pb, pd)
    return b

def example_setup_uc():

    
    f = 2 * np.eye(1)
    q = 1 * np.eye(1)
    h = np.array([[1.]])
    r = np.eye(1)

    pb = 0.01
    ps = 0.99
    pd = 0.6
    
    birth_model = [
            GaussianComponent(1.0, np.zeros(1), np.eye(1)),
            GaussianComponent(1.0, 1 + np.zeros(1), np.eye(1)),
    ]
    clutter_mean = np.zeros(1)
    clutter_var = np.eye(1)
    llambda = 2
    
    b = BerGSFUniformClutter(f, q, h, r, birth_model, llambda, [(-2, 2)], ps, pb, pd)
    return b

example_setup = example_setup_uc

__doc__ = bergsf.__doc__
if hasattr(bergsf, "__all__"):
    __all__ = bergsf.__all__
