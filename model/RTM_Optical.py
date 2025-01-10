# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:44:43 2022

@author: hliu
"""
from numpy import pi
import numpy as np
from RTM_initial import CIxy, weighted_sum_over_lidf_solar_vec, calc_extinc_coeff_pars
from RTM_initial import get_gauss_legendre_quadrature, get_conversion_factors


def rtm_o(d, p, lai, hemi_dif_pars, stem_flag=False):
    if stem_flag:
        rho, tau = d.stem[0].flatten(), d.stem[1].flatten()
    else:
        rho, tau = d.leaf[0].flatten(), d.leaf[1].flatten()

    rg = d.soil
    lidf = d.lidf

    CI_flag = p.CI_flag
    CI_thres = p.CI_thres
    CIs, CIo = d.CIs, d.CIo

    ks, ko, kd = d.ks, d.ko, calc_extinc_coeff_pars(CI_flag, CI_thres, lidf, lai)
    sob, sof = d.sob, d.sof
    tts, tto, psi = d.tts, d.tto, d.psi

    # soil and canopy properties
    w = rho + tau  # leaf single scattering albedo

    i0 = max(1 - np.exp(-ks * lai * CIs), 1e-16)
    iv = max(1 - np.exp(-ko * lai * CIo), 1e-16)

    t0 = 1 - i0
    tv = 1 - iv

    dso = define_geometric_constant(tts, tto, psi)

    [tsstoo, sumint] = hotspot_calculations(lai, ko*CIo, ks*CIs, dso)

    [sob_vsla, sof_vsla, kgd] = BRF_hemi_func(d.hemi_pars, d.CIs, d.CIo, lai)

    [sob_vsla_dif, sof_vsla_dif, kg_dif] = BRF_dif_func(d.dif_pars, d.CIs, d.CIo, lai)

    [sob_vsla_hemi_dif, sof_vsla_hemi_dif, kgd_dif] = hemi_dif_pars

    rho_obs = iv / 2 / lai

    iD = i_hemi(CI_flag, lai, lidf, CI_thres)
    td = 1 - iD

    p = 1 - iD / lai

    rho_hemi = iD / 2 / lai
    rho_dif = iv / 2 / lai
    rho_dif_hemi = iD / 2 / lai

    wso = sob * rho + sof * tau

    Tdn = t0 + i0 * w * rho_hemi / (1 - p * w)
    Tup_o = tv + iD * w * rho_obs / (1 - p * w)
    Rdn = iD * w * rho_hemi / (1 - p * w)

    BRFv = wso * lai * CIo * sumint + i0 * w * w * p * rho_obs / (1 - p * w)
    BRFs = tsstoo * rg
    BRFm = rg * Tdn * Tup_o / (1 - rg * Rdn) - t0 * rg * tv
    BRF = BRFv + BRFs + BRFm

    Tup_hemi = td + iD * w * rho_hemi / (1 - p * w)

    # albedo
    Rv = sob_vsla * rho + sof_vsla * tau + i0 * w ** 2 * p * rho_hemi / (1 - p * w)
    Rs = kgd * rg
    Rm = rg * Tdn * Tup_hemi / (1 - rg * Rdn) - t0 * rg * td
    R = Rv + Rs + Rm

    # absorption
    Av = i0 * (1 - w) / (1 - p * w)
    Aup = iD * (1 - w) / (1 - p * w)
    Am = rg * Tdn * Aup / (1 - rg * Rdn)
    A_dir = Av + Am

    Tdn_dif = td + iD * w * rho_dif_hemi / (1 - p * w)
    Tup_difo = tv + iD * w * rho_dif / (1 - p * w)
    Rdn_dif = iD * w * rho_dif_hemi / (1 - p * w)

    BRF_difv = sob_vsla_dif * rho + sof_vsla_dif * tau + iD * w ** 2 * p * rho_dif / (1 - p * w)
    BRF_difs = kg_dif * rg
    BRF_difm = rg * Tdn_dif * Tup_difo / (1 - rg * Rdn_dif) - td * rg * tv
    BRF_dif = BRF_difv + BRF_difs + BRF_difm

    Tup_dif_hemi = td + iD * w * rho_dif_hemi / (1 - p * w)

    R_difv = sob_vsla_hemi_dif * rho + sof_vsla_hemi_dif * tau + iD * w ** 2 * p * rho_dif_hemi / (1 - p * w)
    R_difs = kgd_dif * rg
    R_difm = rg * Tdn_dif * Tup_dif_hemi / (1 - rg * Rdn_dif) - td * rg * td
    R_dif = R_difv + R_difs + R_difm

    # absorption
    Aup_dif = iD * (1 - w) / (1 - p * w)
    A_difv = iD * (1 - w) / (1 - p * w)
    A_difm = rg * Tdn_dif * Aup_dif / (1 - rg * Rdn_dif)
    A_dif = A_difv + A_difm

    # diffuse absorption for soil
    A_dif_soil = (1 - rg) * Tdn_dif * rg / (1 - rg * Rdn_dif)

    out = {'A_dir': A_dir,
           'A_dif': A_dif,
           'A_dif_soil': A_dif_soil,
           'kc': lai * CIo * sumint,
           'kg': tsstoo,
           'kd': kd,
           'BRF': BRF,
           'BRF_dif': BRF_dif,
           'i0': i0,
           'iD': iD,
           'R': R,
           'R_dif': R_dif,
           'Rs': Rs,
           'p': p,
           'tv': tv,
           'tss': np.exp(-ks * lai),
           'rho_obs': rho_obs,
           'rho_hemi': rho_hemi,
           'kgd': kgd,
           'kg_dif': kg_dif,
           'sob_vsla': sob_vsla,
           'sof_vsla': sof_vsla,
           'sob_vsla_dif': sob_vsla_dif,
           'sof_vsla_dif': sof_vsla_dif,
           }

    return out


def hotspot_calculations(lai, ko, ks, dso):
    # Treatment of the hotspot-effect
    alf = 1e36

    hotspot = 0.01  # hotspot parameter, approximately equal to the ratio leaf width/canopy height

    tss = np.exp(-ks * lai)

    # Apply correction 2/(K+k) suggested by F.-M. Breon
    if hotspot > 0.:
        alf = (dso / hotspot) * 2. / (ks + ko)
    if alf == 0.:
        # The pure hotspot
        tsstoo = tss
        sumint = (1. - tss) / (ks * lai)
    else:
        # Outside the hotspot
        alf = (dso / hotspot) * 2. / (ks + ko)
        fhot = lai * np.sqrt(ko * ks)
        # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal partitioning of the slope of the joint probability function
        x1 = 0.
        y1 = 0.
        f1 = 1.
        fint = (1. - np.exp(-alf)) * .05
        sumint = 0.
        for istep in range(1, 21):
            if istep < 20:
                x2 = -np.log(1. - istep * fint) / alf
            else:
                x2 = 1.
            y2 = -(ko + ks) * lai * x2 + fhot * (1. - np.exp(-alf * x2)) / alf
            f2 = np.exp(y2)
            sumint = sumint + (f2 - f1) * (x2 - x1) / (y2 - y1)
            x1 = x2
            y1 = y2
            f1 = f2

        tsstoo = f1
        if np.isnan(sumint):
            sumint = 0.

    return tsstoo, sumint


def hotspot_calculations_vec(lai, ko, ks, dso):

    hotspot = np.full(ko.shape, 0.01)

    tss = np.exp(-ks * lai)

    tsstoo = np.zeros(tss.shape)
    sumint = np.zeros(lai.shape)

    # Treatment of the hotspot-effect
    alf = np.ones(lai.shape) * 1e36
    alf[hotspot > 0] = (dso[hotspot > 0] / hotspot[hotspot > 0]) * 2. / (ks[hotspot > 0] + ko[hotspot > 0])

    index = np.logical_and(lai > 0, alf == 0)
    # The pure hotspot
    tsstoo[index] = tss[index]
    sumint[index] = (1. - tss[index]) / (ks[index] * lai[index])

    # Outside the hotspot
    index = np.logical_and(lai > 0, alf != 0)
    fhot = lai[index] * np.sqrt(ko[index] * ks[index])
    # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal partitioning of the slope of the joint probability function
    x1 = np.zeros(fhot.shape)
    y1 = np.zeros(fhot.shape)
    f1 = np.ones(fhot.shape)
    fint = (1. - np.exp(-alf[index])) * .05
    for istep in range(1, 21):
        if istep < 20:
            x2 = -np.log(1. - istep * fint) / alf[index]
        else:
            x2 = np.ones(fhot.shape)
        y2 = -(ko[index] + ks[index]) * lai[index] * x2 + fhot * (1. - np.exp(-alf[index] * x2)) / alf[index]
        f2 = np.exp(y2)
        sumint[index] = sumint[index] + (f2 - f1) * (x2 - x1) / (y2 - y1)
        x1 = np.copy(x2)
        y1 = np.copy(y2)
        f1 = np.copy(f2)

    tsstoo[index] = f1
    sumint[np.isnan(sumint)] = 0.

    return tsstoo, sumint


def define_geometric_constant(tts, tto, psi):
    tants = np.tan(np.radians(tts))
    tanto = np.tan(np.radians(tto))
    cospsi = np.cos(np.radians(psi))
    dso = np.sqrt(tants ** 2. + tanto ** 2. - 2. * tants * tanto * cospsi)
    return dso


def BRF_hemi_func(pars, CIs, CIo, lai):
    # Nodes and weights of Gauss-Legendre integrals
    xx, ww = get_gauss_legendre_quadrature()

    conv1_tL, conv2_tL = get_conversion_factors(0.0, np.pi / 2.0)
    conv1_pL, conv2_pL = get_conversion_factors(0.0, 2.0 * pi)

    [tts, tto, psi, ks, ko, sob, sof] = pars
    dso = define_geometric_constant(tts, tto, psi)

    lai_arr = np.full(tts.shape, lai)
    tsstoo, sumint = hotspot_calculations_vec(lai_arr, ko*CIo, ks*CIs, dso)

    k1 = (sob * sumint * CIo * lai_arr / pi).reshape(8, 8)
    k2 = (sof * sumint * CIo * lai_arr / pi).reshape(8, 8)
    k3 = (tsstoo / pi).reshape(8, 8)

    neword_tL = conv1_tL * xx + conv2_tL
    mu_tL = np.cos(neword_tL)
    sin_tL = np.sin(neword_tL)

    ww1 = ww * conv1_pL * mu_tL * sin_tL
    ww2 = ww * conv1_tL

    sob_vsla = np.einsum('ij,i,j->', k1, ww1, ww2)
    sof_vsla = np.einsum('ij,i,j->', k2, ww1, ww2)
    kgd = np.einsum('ij,i,j->', k3, ww1, ww2)

    return sob_vsla, sof_vsla, kgd


def BRF_dif_func(pars, CIs, CIo, lai):
    # Nodes and weights of Gauss-Legendre integrals
    xx, ww = get_gauss_legendre_quadrature()

    conv1_tL, conv2_tL = get_conversion_factors(0.0, np.pi / 2.0)
    conv1_pL, conv2_pL = get_conversion_factors(0.0, 2.0 * pi)

    [tta, tto, psi, ks, ko, sob, sof] = pars
    dso = define_geometric_constant(tta, tto, psi)

    lai_arr = np.full(tta.shape, lai)
    tsstoo, sumint = hotspot_calculations_vec(lai_arr, ko*CIo, ks*CIs, dso)

    k1 = (sob * sumint * CIo * lai_arr / pi).reshape(8, 8)
    k2 = (sof * sumint * CIo * lai_arr / pi).reshape(8, 8)
    k3 = (tsstoo / pi).reshape(8, 8)

    neword_tL = conv1_tL * xx + conv2_tL
    mu_tL = np.cos(neword_tL)
    sin_tL = np.sin(neword_tL)

    ww1 = ww * conv1_pL * mu_tL * sin_tL
    ww2 = ww * conv1_tL

    sob_vsla = np.einsum('ij,i,j->', k1, ww1, ww2)
    sof_vsla = np.einsum('ij,i,j->', k2, ww1, ww2)
    kgd = np.einsum('ij,i,j->', k3, ww1, ww2)

    return sob_vsla, sof_vsla, kgd


def BRF_hemi_dif_func(pars, CIs, CIo, lai):
    # Nodes and weights of Gauss-Legendre integrals
    xx, ww = get_gauss_legendre_quadrature()

    conv1_mL, conv2_mL = get_conversion_factors(0.0, np.pi / 2.0)
    conv1_nL, conv2_nL = get_conversion_factors(0.0, 2.0 * pi)

    conv1_tL, conv2_tL = get_conversion_factors(0.0, np.pi / 2.0)
    conv1_pL, conv2_pL = get_conversion_factors(0.0, 2.0 * pi)

    [tts, tto, psi, ks, ko, sob, sof] = pars
    dso = define_geometric_constant(tts, tto, psi)

    lai_arr = np.full(tts.shape, lai)
    tsstoo, sumint = hotspot_calculations_vec(lai_arr, ko*CIo, ks*CIs, dso)

    k1 = (sob * sumint * CIo * lai_arr / pi).reshape(8, 8, 8, 8)
    k2 = (sof * sumint * CIo * lai_arr / pi).reshape(8, 8, 8, 8)
    k3 = (tsstoo / pi).reshape(8, 8, 8, 8)

    neword_tL = conv1_tL * xx + conv2_tL
    mu_tL = np.cos(neword_tL)
    sin_tL = np.sin(neword_tL)

    neword_mL = conv1_mL * xx + conv2_mL
    mu_mL = np.cos(neword_mL)
    sin_mL = np.sin(neword_mL)

    ww1 = ww * conv1_pL * mu_tL * sin_tL
    ww2 = ww * conv1_tL / pi
    ww3 = ww * conv1_nL * mu_mL * sin_mL
    ww4 = ww * conv1_mL

    sob_vsla = np.einsum('ijkl,i,j,k,l->', k1, ww1, ww2, ww3, ww4)
    sof_vsla = np.einsum('ijkl,i,j,k,l->', k2, ww1, ww2, ww3, ww4)
    kgd_dif = np.einsum('ijkl,i,j,k,l->', k3, ww1, ww2, ww3, ww4)

    return sob_vsla, sof_vsla, kgd_dif


def i_hemi(CI_flag, lai, lidf, CI_thres):
    # Nodes and weights of Gauss-Legendre integrals
    xx, ww = get_gauss_legendre_quadrature()

    conv1_tL, conv2_tL = get_conversion_factors(0.0, np.pi / 2.0)

    neword_tL = conv1_tL * xx + conv2_tL
    mu_tL = np.cos(neword_tL)
    sin_tL = np.sin(neword_tL)

    tta = neword_tL * 180 / np.pi
    Ga, ka = weighted_sum_over_lidf_solar_vec(tta, lidf)
    CIa = CIxy(CI_flag, tta, CI_thres)

    ia = 1 - np.exp(-ka * lai * CIa)

    sum_tL = np.sum(ww * mu_tL * sin_tL * ia * 2) * conv1_tL

    return sum_tL
