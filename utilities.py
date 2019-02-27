# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 19:58:51 2018

@author: Kin Ian Lo
"""
import numpy as np
from scipy.interpolate import interp1d

def get_VN_entropy(bloch_lengths):
    """
    Gives the von Neumann entropies for an array of Bloch lengths
    """
    lam = (1-bloch_lengths)/2
    vne = np.nan_to_num(-lam * np.log(lam) - (1-lam) * np.log((1-lam)))
    return vne

def get_bloch_length(VN_entropies):
    """
    Gives the Bloch lengths for an array of von Neumann entropies
    """
    V = np.linspace(1, 0, 1000000)
    E = get_VN_entropy(V)
    f = interp1d(E, V, kind='cubic', assume_sorted=True, bounds_error=False, fill_value=(1, 0))
    return f(VN_entropies)
