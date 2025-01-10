# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:39:26 2022

@author: hliu
"""
from constants import T2K, RGAS, R, Rhoa, Mair
from scipy import optimize
import numpy as np

from Ebal_funcs import calc_vapor_pressure

# from sympy import Symbol, nsolve


def PhotoSynth_Jen(meteo, p):
    Q = meteo[0]  # [umol m-2 s-1] absorbed PAR flux
    Cs = meteo[1] * p.ppm2bar
    T = meteo[2] + T2K
    eb = meteo[3]

    # Calculate temp dependancies of Michaelis–Menten constants for CO2, O2
    Km = calc_michaelis_menten_constants(p, T)

    # Effect of temp on CO2 compensation point
    Gamma_star = arrh(p.Gamma_star25, p.Eag, T)

    # Calculations at 25 degrees C or the measurement temperature
    Rd = peaked_arrh(p.Rd25, p.Ear, T, p.deltaSr, p.Hdr)  # calc_resp(p.Rd25, p.Ear, T)

    # Calculate temperature dependancies on Vcmax and Jmax
    Vcmax = peaked_arrh(p.Vcmax25, p.Eav, T, p.deltaSv, p.Hdv)

    # Calculate cytochrome b6f complex
    # kq = p.kq * exp(p.Eaq / R * (1 / p.Tref - 1 / T))  # Cyt b6f kcat for PQH2, s-1
    Vqmax = p.CB6F * p.kq  # Max Cyt b6f activity, mol e- m-2 s-1

    # Calculate electron transportation rate
    JP700_j = Q * Vqmax / (Q + Vqmax / (p.a1 * (p.Kp1 / (p.Kp1 + p.Kd + p.Kf))))  #

    RH = min(1, eb / calc_vapor_pressure(T-273.15))

    A_Pars = [Vcmax, Km, JP700_j, p.nl, p.nc, Gamma_star, Rd]
    Cm_Pars = [p.gm, Cs, RH, p.minCi, p.BallBerrySlope, p.BallBerry0, p.ppm2bar]

    try:
        Cm = optimize.brentq(opt_Cm, -1, 1, (A_Pars, Cm_Pars), 1E-6)
    except:
        # print("Can't find the intercellular CO2 concentration!")
        Cm = Cs

    A, Ag, Aj, Ac = Compute_A(Cm, Vcmax, Km, JP700_j, p.nl, p.nc, Gamma_star, Rd)

    Ci = Cm + A * p.ppm2bar / p.gm
    gs = max(0.01, 1.6 * A * p.ppm2bar / max(Cs - Ci, 1E-6))  # stomatal conductance
    rcw = (Rhoa / (Mair * 1E-3)) / gs  # stomatal resistance

    JP700_a, JP680_a = calc_ETR(p, Cm, Ac, JP700_j, Gamma_star)

    fqe2, fqe1 = calc_fluorescence_activation(p, Q, JP700_a, JP680_a, JP700_j, p.kq)

    return rcw, Ci / p.ppm2bar, Ag, A, fqe2, fqe1


def calc_fluorescence_activation(p, Q, JP700_a, JP680_a, JP700_j, kq):
    """
    Rubisco carboxylation limited rate of photosynthesis
    
    Parameters
    ----------
    C : float 
        CO2 concentration in mesophyll cell 
    Gamma_star : float
        CO2 compensation point
    """
    phi1P_max = p.Kp1 / (p.Kp1 + p.Kd + p.Kf)  # Maximum photochemical yield PS I

    CB6F_a = JP700_j / kq  # Eqns. 21, 30a, 34
    phi1P_a = JP700_a / (Q * p.a1)  # Eqn. 20
    q1_a = phi1P_a / phi1P_max  # Eqn. 19a
    phi2P_a = JP680_a / (Q * p.a2)  # Eqn. 26
    q2_a = 1 - CB6F_a / p.CB6F  # Eqns. 28 and 34

    # N.B., rearrange Eqn. 25a to solve for Kn2_a
    Kn2_a = ((p.Kp2 ** 2. * phi2P_a ** 2 - 2 * p.Kp2 ** 2 * phi2P_a * q2_a + \
              p.Kp2 ** 2 * q2_a ** 2 - 4 * p.Kp2 * p.Ku2 * phi2P_a ** 2 * q2_a + \
              2 * p.Kp2 * p.Ku2 * phi2P_a ** 2 + 2 * p.Kp2 * p.Ku2 * phi2P_a * q2_a + \
              p.Ku2 ** 2 * phi2P_a ** 2) ** 0.5 - p.Kp2 * phi2P_a + p.Ku2 * phi2P_a + \
             p.Kp2 * q2_a) / (2 * phi2P_a) - p.Kf - p.Ku2 - p.Kd

    # Photosystem II (Eqns. 23a-23e and 25a-25d)
    phi2p_a = q2_a * p.Kp2 / (p.Kp2 + Kn2_a + p.Kd + p.Kf + p.Ku2)
    phi2f_a = q2_a * p.Kf / (p.Kp2 + Kn2_a + p.Kd + p.Kf + p.Ku2) + (1 - q2_a) * p.Kf / (Kn2_a + p.Kd + p.Kf + p.Ku2)
    phi2u_a = q2_a * p.Ku2 / (p.Kp2 + Kn2_a + p.Kd + p.Kf + p.Ku2) + (1 - q2_a) * p.Ku2 / (Kn2_a + p.Kd + p.Kf + p.Ku2)

    phi2P_a = phi2p_a / (1 - phi2u_a)
    phi2F_a = phi2f_a / (1 - phi2u_a)

    # For Photosystem I (Eqns. 19a-19d)
    phi1P_a = q1_a * p.Kp1 / (p.Kp1 + p.Kd + p.Kf)
    phi1F_a = q1_a * p.Kf / (p.Kp1 + p.Kd + p.Kf) + (1 - q1_a) * p.Kf / (p.Kn1 + p.Kd + p.Kf)

    # PAM measured fluorescence levels (Eqns. 38-42)
    #   N.B., hardcoding of a2(1) for dark-adapted value
    # Fo_a2 = p.a2 * p.Kf / (p.Kp2 + p.Kd + p.Kf) * p.eps2
    # Fo_a1 = p.a1 * p.Kf / (p.Kp1 + p.Kd + p.Kf) * p.eps1

    # Fs_a2 = p.a2 * phi2F_a * p.eps2
    # Fs_a1 = p.a1 * phi1F_a * p.eps1

    Fo_a2 = p.Kf / (p.Kp2 + p.Kd + p.Kf)
    Fo_a1 = p.Kf / (p.Kp1 + p.Kd + p.Kf)

    Fs_a2 = phi2F_a
    Fs_a1 = phi1F_a

    """
    Estimating photosynthetic capacity from leaf reflectance and Chl fluorescence 
    by coupling radiative transfer to a model for photosynthesis
    https://nph.onlinelibrary.wiley.com/doi/full/10.1111/nph.15782
    """
    fqe2 = Fs_a2 / Fo_a2
    fqe1 = Fs_a1 / Fo_a1

    return fqe2, fqe1


def calc_ETR(p, C, Ac, JP700_j, Gamma_star):
    """
    Rubisco carboxylation limited rate of photosynthesis
    
    Parameters
    ----------
    C : float 
        CO2 concentration in mesophyll cell 
    Gamma_star : float
        CO2 compensation point
    """

    eta = (1 - (p.nl / p.nc) + (3 + 7 * Gamma_star / C) / ((4 + 8 * Gamma_star / C) * p.nc))  # PS I/II ETR

    JP680_c = Ac * 4 * (1 + 2 * Gamma_star / C) / (1 - Gamma_star / C)
    JP700_c = JP680_c * eta

    JP680_j = JP700_j / eta

    theta_hyperbol = 1.0

    # Select minimum PS1 ETR
    JP700_a = sel_root(theta_hyperbol, -(JP700_c + JP700_j), JP700_c * JP700_j, np.sign(-JP700_c))

    # Select minimum PS2 ETR
    JP680_a = sel_root(theta_hyperbol, -(JP680_c + JP680_j), JP680_c * JP680_j, np.sign(-JP680_c))

    return JP700_a, JP680_a


def opt_Cm(x0, A_Pars, Cm_Pars):
    [Vcmax_m, Km, JP700_mj, nl, nc, Gamma_star, Rd] = A_Pars
    A, _, _, _ = Compute_A(x0, Vcmax_m, Km, JP700_mj, nl, nc, Gamma_star, Rd)

    [gm, Cs, RH, minCi, BallBerrySlope, BallBerry0, ppm2bar] = Cm_Pars
    Ci = BallBerry(Cs, RH, A * ppm2bar, BallBerrySlope, BallBerry0, minCi)
    x1 = Ci - A * ppm2bar / gm

    return x0 - x1


def Compute_A(Cm, Vcmax, Km, J700, nl, nc, Gamma_star, Rd):
    """
    Parameters
    ----------
    theta_hyperbol : float
        Curvature of the light response.
        See Peltoniemi et al. 2012 Tree Phys, 32, 510-519

    Reference:
        Steady-state models of photosynthesis, 2013
        https://onlinelibrary.wiley.com/doi/pdf/10.1111/pce.12098
    """
    theta_hyperbol = 1.0

    # Rubisco carboxylation limited rate of photosynthesis
    Ac = Vcmax * (Cm - Gamma_star) / (Km + Cm)

    # Light-limited rate of photosynthesis allowed by RuBP regeneration
    Je = J700 * (1 - nl / nc + (3 * Cm + 7 * Gamma_star) / ((8 * Gamma_star + 4 * Cm) * nl))
    Aj = Je / 4.0 * ((Cm - Gamma_star) / (2 * Gamma_star + Cm))

    Ag = sel_root(theta_hyperbol, -(Ac + Aj), Ac * Aj, -1)

    A = Ag - Rd

    return A, Ag, Aj, Ac


def BallBerry(Cs, RH, A, BallBerrySlope, BallBerry0, minCi):
    if BallBerry0 == 0:
        Ci = max(minCi * Cs, Cs * (1 - 1.6 / (BallBerrySlope * RH)))

    else:
        gs = max(BallBerry0, BallBerrySlope * A * RH / (Cs + 1E-9) + BallBerry0)
        Ci = max(minCi * Cs, Cs - 1.6 * A / gs)

    return Ci


def calc_resp(Rd25, Ear, T):
    """ Calculate leaf respiration accounting for temperature dependence.

    Parameters:
    ----------
    Rd25 : float
        Estimate of respiration rate at the reference temperature 25 deg C
        or 298 K
    Tref : float
        reference temperature
    Q10 : float
        ratio of respiration at a given temperature divided by respiration
        at a temperature 10 degrees lower
    Ear : float
        activation energy for the parameter [J mol-1]
    Returns:
    -------
    Rt : float
        leaf respiration

    References:
    -----------
    Tjoelker et al (2001) GCB, 7, 223-230.
    """
    Rd = arrh(Rd25, Ear, T)

    return Rd


def sel_root(a, b, c, dsign):
    """    
    quadratic formula, root of least magnitude
    """
    #  sel_root - select a root based on the fourth arg (dsign = discriminant sign)
    #    for the eqn ax^2 + bx + c,
    #    if dsign is:
    #       -1, 0: choose the smaller root
    #       +1: choose the larger root
    #  NOTE: technically, we should check a, but in biochemical, a is always > 0, dsign is always not equal to 0
    if a == 0:  # note: this works because 'a' is a scalar parameter!
        x = -c / b
    else:
        x = (-b + dsign * np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    return x


def calc_michaelis_menten_constants(p, Tleaf):
    """ Michaelis-Menten constant for O2/CO2, Arrhenius temp dependancy
    Parameters:
    ----------
    Tleaf : float
        leaf temperature [deg K]

    Returns:
    Km : float

    """
    Kc = arrh(p.Kc25, p.Ec, Tleaf)
    Ko = arrh(p.Ko25, p.Eo, Tleaf)

    Km = Kc * (1.0 + p.O_c3 / Ko)

    return Km


def arrh(k25, Ea, Tk):
    """ Temperature dependence of kinetic parameters is described by an
    Arrhenius function.

    Parameters:
    ----------
    k25 : float
        rate parameter value at 25 degC or 298 K
    Ea : float
        activation energy for the parameter [J mol-1]
    Tk : float
        leaf temperature [deg K]

    Returns:
    -------
    kt : float
        temperature dependence on parameter

    References:
    -----------
    * Medlyn et al. 2002, PCE, 25, 1167-1179.
    """
    return k25 * np.exp((Ea * (Tk - 298.15)) / (298.15 * RGAS * Tk))


def peaked_arrh(k25, Ea, Tk, deltaS, Hd):
    """ Temperature dependancy approximated by peaked Arrhenius eqn,
    accounting for the rate of inhibition at higher temperatures.

    Parameters:
    ----------
    k25 : float
        rate parameter value at 25 degC or 298 K
    Ea : float
        activation energy for the parameter [J mol-1]
    Tk : float
        leaf temperature [deg K]
    deltaS : float
        entropy factor [J mol-1 K-1)
    Hd : float
        describes rate of decrease about the optimum temp [J mol-1]

    Returns:
    -------
    kt : float
        temperature dependence on parameter

    References:
    -----------
    * Medlyn et al. 2002, PCE, 25, 1167-1179.

    """
    arg1 = arrh(k25, Ea, Tk)
    arg2 = 1.0 + np.exp((298.15 * deltaS - Hd) / (298.15 * RGAS))
    arg3 = 1.0 + np.exp((Tk * deltaS - Hd) / (Tk * RGAS))

    return arg1 * arg2 / arg3
