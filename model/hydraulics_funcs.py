# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:42:56 2023

@author: hliu
"""
from math import log10, exp

def saturated_matrix_potential(Sand):
    """
    %Sand :
        Sand : 92
        Loam : 43
        Clay : 22

    Returns
    -------
    phis_sat: the matrix potential at saturation Osat (mm)

    """
    
    return -10*10**(1.88-0.0131*Sand)

def calc_b(Clay):
    """
    %Clay :
        Sand : 3
        Loam : 18
        Clay : 58

    Returns
    -------
    b: to %sand and %clay

    """
    
    return 2.91 + 0.159*Clay

def cal_thetas(SOC):
    """
    Parameters
    ----------
    SOC: list 
        Soil composition, 1 by 3, percentages of sand, silt and clay

    Returns
    -------
    thetas: float
        water content at saturation
        
    Reference:
    Saxton, K. E., et al. "Estimating generalized soil-water characteristics
    from texture." Soil Science Society of America Journal 50.4 (1986): 1031-1036.
    
    """
    
    S = SOC[0] # [%] percentage of sand 
    C = SOC[2] # [%] percentage of clay

    thetas = 0.332 - 7.25E-4*S + 0.1276*log10(C) # [m³/m³] water content at saturation
    
    return thetas

def hygroscopic_point(SOC):
    """
    Parameters
    ----------
    SOC: list 
        Soil composition, 1 by 3, percentages of sand, silt and clay

    Returns
    -------
    wp: float
        wilting point, unitless

    Reference:
    Saxton, K. E., et al. "Estimating generalized soil-water characteristics
    from texture." Soil Science Society of America Journal 50.4 (1986): 1031-1036.
    
    """
    
    S = SOC[0] # [%] percentage of sand 
    C = SOC[2] # [%] percentage of clay
    
    thetas = 0.332 - 7.25E-4*S + 0.1276*log10(C) # [m³/m³] water content at saturation

    A = exp(-4.396 - 0.0715*C + -4.880E-4*S**2 -4.285E-5*S**2*C)*100 #eq 5
    B = -3.140 - 2.22E-3*C**2 - 3.484E-5*S**2 - 3.484E-5*S**2*C - 3.484E-5*S**2*C #eq 6
    
    hp = (10000/A)**(1/B)/thetas # [] hygroscopic point
    
    return hp

def field_capacity(SOC):
    """
    Parameters
    ----------
    SOC: list 
        Soil composition, 1 by 3, percentages of sand, silt and clay

    Returns
    -------
    fc: float
        field capacity, unitless

    Reference:
    Saxton, K. E., et al. "Estimating generalized soil-water characteristics
    from texture." Soil Science Society of America Journal 50.4 (1986): 1031-1036.
    
    """
    
    S = SOC[0] # [%] percentage of sand 
    C = SOC[2] # [%] percentage of clay
    
    thetas = 0.332 - 7.25E-4*S + 0.1276*log10(C) # [m³/m³] water content at saturation

    A = exp(-4.396 - 0.0715*C + -4.880E-4*S**2 -4.285E-5*S**2*C)*100 #eq 5
    B = -3.140 - 2.22E-3*C**2 - 3.484E-5*S**2 - 3.484E-5*S**2*C - 3.484E-5*S**2*C #eq 6
    
    fc = (10/A)**(1/B)/thetas # [] field capacity
    
    return fc
  