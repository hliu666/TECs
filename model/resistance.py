# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:36:30 2022

@author: 16072
"""
from numpy import log, arctan, pi, exp, sinh
import numpy as np
from constants import KARMAN, U_FRICTION_MIN


def calc_Psi_M(zol):
    if zol > 0:
        # for stable and netural (zoL = 0 -> Psi_M = 0) conditions
        a = 6.1
        b = 2.5
        psi_m = -a * np.log(zol + (1.0 + zol**b)**(1.0 / b))
    else:
        # for unstable conditions
        y = -zol
        a = 0.33
        b = 0.41
        x = (y / a)**0.333333
    
        psi_0 = -np.log(a) + 3**0.5 * b * a**0.333333 * np.pi / 6.0
        y = np.minimum(y, b**-3)
        psi_m = (np.log(a + y) - 3.0 * b * y**0.333333 +
                    (b * a**0.333333) / 2.0 * np.log((1.0 + x)**2 / (1.0 - x + x**2)) +
                    3.0**0.5 * b * a**0.333333 * np.arctan((2.0 * x - 1.0) / 3**0.5) +
                    psi_0)

    return psi_m


def calc_u_star(u, z_u, L, d_0, z_0M):
    '''Friction velocity.

    Parameters
    ----------
    u : float
        wind speed above the surface (m s-1).
    z_u : float
        wind speed measurement height (m).
    L : float
        Monin Obukhov stability length (m).
    d_0 : float
        zero-plane displacement height (m).
    z_0M : float
        aerodynamic roughness length for momentum transport (m).

    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
        Cambridge: Cambridge University Press.
    '''

    # calculate correction factors in other conditions
    L = max(L, 1e-36)
    Psi_M = calc_Psi_M((z_u - d_0) / L)
    Psi_M0 = calc_Psi_M(z_0M / L)
    del L
    u_star = u * KARMAN / (np.log((z_u - d_0) / z_0M) - Psi_M + Psi_M0)
    return max(u_star, U_FRICTION_MIN)


def calc_u_C_star(u_friction, h_C, d_0, z_0M, L=float('inf')):
    ''' MOST wind speed at the canopy

    Parameters
    ----------
    u_friction : float
        friction velocity (m s-1).
    h_C : float
        canopy height (m).
    d_0 : float
        zero-plane displacement height.
    z_0M : float
        aerodynamic roughness length for momentum transport (m).
    L : float, optional
        Monin-Obukhov length (m).

    Returns
    -------
    u_C : float
        wind speed at the canop interface (m s-1).
    '''

    Psi_M = calc_Psi_M((h_C - d_0) / L)
    Psi_M0 = calc_Psi_M(z_0M / L)

    # calcualte u_C, wind speed at the top of (or above) the canopy
    u_C = (u_friction * (np.log((h_C - d_0) / z_0M) - Psi_M + Psi_M0)) / KARMAN
    return u_C


def calc_R_x_Choudhury(p, u_C, F, leaf_width, alpha_prime=3.0):
    """ Estimates aerodynamic resistance at the canopy boundary layer.

    Estimates the aerodynamic resistance at the canopy boundary layer based on the
    K-Theory model of [Choudhury1988]_.

    Parameters
    ----------
    u_C : float
        wind speed at the canopy interface (m s-1).
    F : float
        local Leaf Area Index.
    leaf_width : float
        efective leaf width size (m).
    alpha_prime : float, optional
        Wind exctinction coefficient, default=3.

    Returns
    -------
    R_x : float
        Aerodynamic resistance at the canopy boundary layer (s m-1).

    References
    ----------
    .. [Choudhury1988] Choudhury, B. J., & Monteith, J. L. (1988). A four-layer model
        for the heat budget of homogeneous land surfaces.
        Royal Meteorological Society, Quarterly Journal, 114(480), 373-398.
        http://dx/doi.org/10.1002/qj.49711448006.
    """

    # Eqs. 29 & 30 [Choudhury1988]_
    R_x = (1.0 / (F * (2.0 * p.CM_a / alpha_prime)
           * np.sqrt(u_C / leaf_width) * (1.0 - np.exp(-alpha_prime / 2.0))))
    # R_x=(alpha_u*(sqrt(leaf_width/U_C)))/(2.0*alpha_0*LAI*(1.-exp(-alpha_u/2.0)))
    return R_x


def calc_R_S_Choudhury(u_star, h_C, z_0M, d_0, zm, z0_soil=0.01, alpha_k=2.0):
    ''' Aerodynamic resistance at the  soil boundary layer.

    Estimates the aerodynamic resistance at the  soil boundary layer based on the
    K-Theory model of [Choudhury1988]_.

    Parameters
    ----------
    u_star : float
        friction velocity (m s-1).
    h_C : float
        canopy height (m).
    z_0M : float
        aerodynamic roughness length for momentum trasport (m).
    d_0 : float
        zero-plane displacement height (m).
    zm : float
        height on measurement of wind speed (m).
    z0_soil : float, optional
        roughness length of the soil layer, use z0_soil=0.01.
    alpha_k : float, optional
        Heat diffusion coefficient, default=2.

    Returns
    -------
    R_S : float
        Aerodynamic resistance at the  soil boundary layer (s m-1).

    References
    ----------
    .. [Choudhury1988] Choudhury, B. J., & Monteith, J. L. (1988). A four-layer model
        for the heat budget of homogeneous land surfaces.
        Royal Meteorological Society, Quarterly Journal, 114(480), 373-398.
        http://dx/doi.org/10.1002/qj.49711448006.
    '''

    # Soil resistance eqs. 24 & 25 [Choudhury1988]_
    K_h = KARMAN * u_star * (h_C - d_0)
    del u_star
    R_S = ((h_C * np.exp(alpha_k) / (alpha_k * K_h))
           * (np.exp(-alpha_k * z0_soil / h_C) - np.exp(-alpha_k * (d_0 + z_0M) / h_C)))

    return R_S


def resistance(p, lai, L, u):

    u_star = calc_u_star(u, p.z_u, L, p.d_0, p.z_0M)
    u_C = calc_u_C_star(u_star, p.h_C, p.d_0, p.z_0M)
    R_x = calc_R_x_Choudhury(p, u_C, lai, p.leaf_width)
    R_s = calc_R_S_Choudhury(u_star, p.h_C, p.z_0M, p.d_0, p.zm)
   
    return u_star, R_x, R_s

