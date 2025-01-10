# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 14:59:26 2022

@author: hliu
"""
import numpy as np
from constants import R_d, epsilon, c_pd, c_pv


def calc_vapor_pressure(T_C):
    """Calculate the saturation water vapour pressure.

    Parameters
    ----------
    T_C : float
        temperature (C).

    Returns
    -------
    ea : float
        saturation water vapour pressure (mb).
    """

    ea = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    return ea


def calc_delta_vapor_pressure(T_C):
    """Calculate the slope of saturation water vapour pressure.

    Parameters
    ----------
    T_C : float
        temperature (C).

    Returns
    -------
    s : float
        slope of the saturation water vapour pressure (kPa K-1)
    """

    s = 4098.0 * (0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))) / ((T_C + 237.3)**2)
    return s*10 #convert Kpa to hpa


def calc_lambda(T_A_K):
    """Calculates the latent heat of vaporization.

    Parameters
    ----------
    T_A_K : float
        Air temperature (Kelvin).

    Returns
    -------
    Lambda : float
        Latent heat of vaporisation (J kg-1).

    References
    ----------
    based on Eq. 3-1 Allen FAO98 """

    Lambda = 1e6 * (2.501 - (2.361e-3 * (T_A_K - 273.15)))
    return Lambda


def calc_rho(p, ea, T_A_K):
    """Calculates the density of air.

    Parameters
    ----------
    p : float
        total air pressure (dry air + water vapour) (mb).
    ea : float
        water vapor pressure at reference height above canopy (mb).
    T_A_K : float
        air temperature at reference height (Kelvin).

    Returns
    -------
    rho : float
        density of air (kg m-3).

    References
    ----------
    based on equation (2.6) from Brutsaert (2005): Hydrology - An Introduction (pp 25)."""

    # p is multiplied by 100 to convert from mb to Pascals
    rho = ((p * 100.0) / (R_d * T_A_K)) * (1.0 - (1.0 - epsilon) * ea / p)
    return rho


def calc_c_p(p, ea):
    """ Calculates the heat capacity of air at constant pressure.

    Parameters
    ----------
    p : float
        total air pressure (dry air + water vapour) (mb).
    ea : float
        water vapor pressure at reference height above canopy (mb).

    Returns
    -------
    c_p : heat capacity of (moist) air at constant pressure (J kg-1 K-1).

    References
    ----------
    based on equation (6.1) from Maarten Ambaum (2010):
    Thermal Physics of the Atmosphere (pp 109)."""

    # first calculate specific humidity, rearanged eq (5.22) from Maarten
    # Ambaum (2010), (pp 100)
    q = epsilon * ea / (p + (epsilon - 1.0) * ea)
    # then the heat capacity of (moist) air
    c_p = (1.0 - q) * c_pd + q * c_pv
    return c_p


               
