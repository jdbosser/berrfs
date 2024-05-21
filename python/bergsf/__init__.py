from .bergsf import *
from .etete import a_fun
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class GaussianComponent: 
    weight: float
    mean: np.ndarray
    cov: np.ndarray

def example_setup():

    
    f = np.eye(2)
    q = np.eye(2)
    h = np.array([[1., 0.]])
    r = np.eye(1)

    pb = 0.01
    ps = 0.99
    pd = 0.6
    
    birth_model = [GaussianComponent(1.0, np.zeros(2), np.eye(2))]
    clutter_mean = np.zeros(1)
    clutter_var = np.eye(1)
    llambda = 2
    
    b = BerGSF(f, q, h, r, birth_model, llambda, clutter_mean, clutter_var, ps, pb, pd)
    print(b)
    return b


__doc__ = bergsf.__doc__
if hasattr(bergsf, "__all__"):
    __all__ = bergsf.__all__
