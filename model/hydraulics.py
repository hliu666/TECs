# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:42:56 2023

@author: hliu
"""
import numpy as np


def calc_beta_e(Soil, sm_top):
    """
    Parameters
    ----------
    Soil : dictionary
        soc_top:
           Soil composition, 1 by 3, percentages of sand, silt and clay 
    sm_top : TYPE
        DESCRIPTION.

    Returns
    -------
    beta_e : float 
        dimensionaless factor that ranges from one when the soil is wet to zero when
        the soil is dry

    """

    theta_sat = Soil["theta_sat"]
    b = Soil["b1"]
    phis_sat = Soil["phis_sat"]

    theta_dry = theta_sat * (-316230 / phis_sat) ** (-1 / b)
    theta_opt = theta_sat * (-158490 / phis_sat) ** (-1 / b)

    beta_e = min((sm_top - theta_dry) / (theta_opt - theta_dry), 1)

    return beta_e


def canopy_storage_capacity(LAI):
    """
    Parameters
    ----------
    LAI: float 
        Leaf area index

    Returns
    -------
    storage: float
        Canopy storage capacity
    """

    storage = LAI * 1E-4 * 1E3  # interception storage, mm

    return storage


def calc_soil_moisture(Soil, sm_top, through_fall, ET):
    """
    Parameters
    ----------
    sm_top : float
        Topsoil moisture (mm)
    through_fall : float
        through_fall (mm/hour).    

    Returns
    -------
    sm_top : float
        Topsoil moisture (mm)
    """

    """
    Top layers soil moisture:
        dSM/dt = P - I - Q - ET
        
        P is precipitation 
        I is interception loss
        ET is evapotranspiration
        Q is surface runoff, when the first layer is fully saturated
    """

    P_I = through_fall / 1000 / 3600  # mm/h -> m/s

    # Evapotranspiration
    if np.isnan(ET):
        ET = 0.0
    else:
        ET = max(ET / 1000, 0)  # convert kg/m2/s -> m/s

    # Update topsoil moisture
    sm_top += (P_I - ET) / (Soil["theta_sat"] * Soil["Zr_top"] * 1) * 3600

    if sm_top > Soil["fc_top"]:
        Q = sm_top - Soil["fc_top"]
        sm_top = sm_top - Q

    elif sm_top < Soil["sh_top"]:
        sm_top = Soil["sh_top"]

    return sm_top


def calc_hy_f(d, p, lai, Ev, ET):
    """
    Parameters
    ----------
    w_can : float
        Canopy water 
    precip : float
        Precipitation (mm/hour).
    Ev : float
        Evaporation (mm/hour).
        
    Returns
    -------
    w_can : float
        Canopy water 
    fwet : float
        wetted fraction of the canopy

    """
    w_can = d.w_can
    precip = d.precip
    sm_top = d.sm_top
    Soil = p.Soil

    Ev = max(Ev * 3600, 0)  # convert kg/m2/s -> mm/hour

    if precip <= 0:
        w_can = min(w_can, 1)
        w_can = max(w_can - Ev, 0)
        through_fall = 0
    else:
        # Canopy interception
        I = canopy_storage_capacity(lai)
        w_can = min(w_can + min(precip, I - w_can), 1)
        w_can = max(w_can - Ev, 0)
        through_fall = max(precip - min(precip, I - w_can), 0)

    fwet = min((w_can / (0.2 * lai)) ** (2 / 3), 1)

    sm_top = calc_soil_moisture(Soil, sm_top, through_fall, ET)

    a = 0.07
    b = 0.29

    if sm_top < a:
        sf = 0
    elif (sm_top >= a) and (sm_top < (a + b) / 2.0):
        sf = 2 * ((sm_top - a) / (b - a)) ** 2
    elif (sm_top > (a + b) / 2.0) and (sm_top < b):
        sf = 1 - 2 * ((sm_top - b) / (b - a)) ** 2
    else:
        sf = 1

    return w_can, fwet, sm_top, sf
