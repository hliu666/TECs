# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:40:17 2022

@author: hliu
"""

def calc_z_0M(h_C):
    """ Aerodynamic roughness lenght.
    Estimates the aerodynamic roughness length for momentum trasport
    as a ratio of canopy height.
    Parameters
    ----------
    h_C : float
        Canopy height (m).
    Returns
    -------
    z_0M : float
        aerodynamic roughness length for momentum transport (m).
    """

    z_0M = h_C * 0.125
    return z_0M


def calc_d_0(h_C):
    ''' Zero-plane displacement height
    Calculates the zero-plane displacement height based on a
    fixed ratio of canopy height.
    Parameters
    ----------
    h_C : float
        canopy height (m).
    Returns
    -------
    d_0 : float
        zero-plane displacement height (m).'''

    d_0 = h_C * 0.65

    return d_0
