# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:57:14 2022

@author: hliu
"""
import numpy as np
from scipy.interpolate import interp1d


# %% 1) SIF at Leaf level
def cal_sif_leaf(p, fluspect_dict, eta_pars):
    [eta2, eta1] = eta_pars

    phi = fluspect_dict['phi']
    wle = fluspect_dict['wle']
    wlp = fluspect_dict['wlp']
    Iwlf = fluspect_dict['Iwlf']

    rho = fluspect_dict['rho']
    tau = fluspect_dict['tau']
    r21 = fluspect_dict['r21']
    t21 = fluspect_dict['t21']

    kChl_iwle = fluspect_dict['kChl_iwle']
    r21_iwle = fluspect_dict['r21_iwle']
    rho_iwle = fluspect_dict['rho_iwle']
    tau_iwle = fluspect_dict['tau_iwle']
    talf_iwle = fluspect_dict['talf_iwle']

    te = fluspect_dict['te']
    tf = fluspect_dict['tf']
    re = fluspect_dict['re']
    rf = fluspect_dict['rf']

    sigmoid = fluspect_dict['sigmoid']

    ndub = p.ndub  # number of doublings applied
    eps = 2 ** (-ndub)

    MbI = eta1 * p.fqe * np.multiply(0.5 * phi[Iwlf][:, None] * eps, kChl_iwle * sigmoid)
    MfI = eta1 * p.fqe * np.multiply(0.5 * phi[Iwlf][:, None] * eps, kChl_iwle * sigmoid)
    MbII = eta2 * p.fqe * np.multiply(0.5 * phi[Iwlf][:, None] * eps, kChl_iwle * sigmoid)
    MfII = eta2 * p.fqe * np.multiply(0.5 * phi[Iwlf][:, None] * eps, kChl_iwle * sigmoid)

    Ih = np.ones((1, len(te)))  # row of ones
    Iv = np.ones((len(tf), 1))  # column of ones

    # Doubling routine
    for i in range(1, ndub):
        xe = te / (1 - re * re)
        ten = te * xe
        ren = re * (1 + ten)

        xf = tf / (1 - rf * rf)
        tfn = tf * xf
        rfn = rf * (1 + tfn)

        A11 = xf[:, None] * Ih + Iv * xe[None, :]
        A12 = (xf[:, None] * xe[None, :]) * (rf[:, None] * Ih + Iv * re[None, :])
        A21 = 1 + (xf[:, None] * xe[None, :]) * (1 + rf[:, None] * re[None, :])
        A22 = (xf[:, None] * rf[:, None]) * Ih + Iv * (xe * re[None, :])

        MbnI = MbI * A21 + MfI * A22
        MfnI = MfI * A11 + MbI * A12
        MbnII = MbII * A21 + MfII * A22
        MfnII = MfII * A11 + MbII * A12

        te = ten
        re = ren
        tf = tfn
        rf = rfn

        MbI = MbnI
        MfI = MfnI
        MbII = MbnII
        MfII = MfnII

    # Here we add the leaf-air interfaces again for obtaining the final
    # leaf level fluorescences.

    g1 = MbI
    f1 = MfI
    g2 = MbII
    f2 = MfII

    Rb = rho + tau ** 2 * r21 / (1 - rho * r21)
    Rb_iwle = interp1d(wlp, Rb.flatten())(wle)

    Xe = Iv * (talf_iwle / (1 - r21_iwle * Rb_iwle)).T
    Xf = np.outer(t21[Iwlf] / (1 - r21[Iwlf] * Rb[Iwlf]), Ih)
    Ye = Iv * (tau_iwle * r21_iwle / (1 - rho_iwle * r21_iwle)).T
    Yf = np.outer(tau[Iwlf] * r21[Iwlf] / (1 - rho[Iwlf] * r21[Iwlf]), Ih)

    A = Xe * (1 + Ye * Yf) * Xf
    B = Xe * (Ye + Yf) * Xf

    g1n = A * g1 + B * f1
    f1n = A * f1 + B * g1
    g2n = A * g2 + B * f2
    f2n = A * f2 + B * g2

    leafopt_MfII = f2n
    leafopt_MbII = g2n
    leafopt_MfI = f1n
    leafopt_MbI = g1n

    return leafopt_MfII, leafopt_MbII, leafopt_MfI, leafopt_MbI


# %% 2) SIF at Canopy level
def cal_rtm_sif(MfI, MbI, netrad_sw_dict, leaf, soil, canopy_pars, dir_pars, hemi_pars, dif_pars, hemi_dif_pars):
    # Define wavelength ranges
    wls = np.arange(400, 851) - 400
    wlf = np.arange(640, 851) - 400
    wle = np.arange(400, 751) - 400

    # Find intersections
    iwlfi = np.intersect1d(wls, wle)
    iwlfo = np.intersect1d(wls, wlf)

    # Get lengths
    nb = len(wls)
    nf = len(iwlfo)

    rho_l, tau_l = leaf
    rs = soil

    rho_l, tau_l = rho_l.flatten(), tau_l.flatten()
    rs = rs.flatten()

    Qins = netrad_sw_dict['Esun_'][0:451]
    Qind = netrad_sw_dict['Esky_'][0:451]

    [i0, iD, p, rho_obs, rho_hemi, tv, kc, kg] = canopy_pars

    [sob, sof, _] = dir_pars
    [sob_vsla, sof_vsla, kgd] = hemi_pars
    [sob_vsla_dif, sof_vsla_dif, kg_dif] = dif_pars
    [sob_vsla_hemi_dif, sof_vsla_hemi_dif, kgd_dif] = hemi_dif_pars

    t0 = 1 - i0
    td = 1 - iD

    wleaf = rho_l + tau_l
    Mf = MfI + MbI

    Qfdir = np.zeros((nb, 11))
    Qfhemi = np.zeros((nb, 11))
    Qapar = np.zeros((nb, 11))
    Qdown = np.zeros((nb, 11))
    Qsig = np.zeros((nb, 12))
    Qfyld = np.zeros((nf, 11))

    Qsig[:, 0] = Qins * i0

    for i in range(11):
        Qapar[:, i] = Qsig[:, i] * (1 - wleaf)
        MQ = Mf @ Qsig[iwlfi, i]

        if i == 0:
            Qfdir[:, i] = Qins * (sob * rho_l + sof * tau_l) * kc
            Qfdir[iwlfo, i] = Qfdir[iwlfo, i] + (MbI @ Qins[iwlfi] * sob + MfI @ Qins[iwlfi] * sof) * kc
            Qfhemi[:, i] = Qins * (sob_vsla * rho_l + sof_vsla * tau_l)
            Qfhemi[iwlfo, i] = Qfhemi[iwlfo, i] + MbI @ Qins[iwlfi] * sob_vsla + MfI @ Qins[iwlfi] * sof_vsla
        else:
            Qfdir[:, i] = Qsig[:, i] * wleaf * rho_obs
            Qfdir[iwlfo, i] = Qfdir[iwlfo, i] + MQ * rho_obs
            Qfhemi[:, i] = Qsig[:, i] * wleaf * rho_hemi
            Qfhemi[iwlfo, i] = Qfhemi[iwlfo, i] + Mf @ Qsig[iwlfi, i] * rho_hemi

        Qfyld[:, i] = MQ
        Qdown[:, i] = Qsig[:, i] * wleaf * rho_hemi
        Qdown[iwlfo, i] = Qdown[iwlfo, i] + MQ * rho_hemi
        Qsig[:, i + 1] = Qsig[:, i] * wleaf * p
        Qsig[iwlfo, i + 1] = Qsig[iwlfo, i + 1] + MQ * p

    Qfhemi_sum = np.sum(Qfhemi, axis=1)
    Qfdir_sum = np.sum(Qfdir, axis=1)

    Qfdir_d = np.zeros((nb, 11))
    Qfhemi_d = np.zeros((nb, 11))
    Qapar_d = np.zeros((nb, 11))
    Qdown_d = np.zeros((nb, 11))
    Qsig_d = np.zeros((nb, 12))
    Qfyld_d = np.zeros((nf, 11))

    Qsig_d[:, 0] = Qind * iD

    for i in range(11):
        Qapar_d[:, i] = Qsig_d[:, i] * (1 - wleaf)
        MQ = Mf @ Qsig_d[iwlfi, i]

        if i == 0:
            Qfdir_d[:, i] = Qind * (rho_l * sob_vsla_dif + tau_l * sof_vsla_dif)
            Qfdir_d[iwlfo, i] = Qfdir_d[iwlfo, i] + MbI @ Qind[iwlfi] * sob_vsla_dif + MfI @ Qind[iwlfi] * sof_vsla_dif
            Qfhemi_d[:, i] = Qind * (rho_l * sob_vsla_hemi_dif + tau_l * sof_vsla_hemi_dif)
            Qfhemi_d[iwlfo, i] = Qfhemi_d[iwlfo, i] + MbI @ Qind[iwlfi] * sob_vsla_hemi_dif + MfI @ Qind[iwlfi] * sof_vsla_hemi_dif
        else:
            Qfdir_d[:, i] = Qsig_d[:, i] * wleaf * rho_obs
            Qfdir_d[iwlfo, i] = Qfdir_d[iwlfo, i] + MQ * rho_obs
            Qfhemi_d[:, i] = Qsig_d[:, i] * wleaf * rho_hemi
            Qfhemi_d[iwlfo, i] = Qfhemi_d[iwlfo, i] + MQ * rho_hemi

        Qfyld_d[:, i] = MQ
        Qdown_d[:, i] = Qsig_d[:, i] * wleaf * rho_hemi
        Qdown_d[iwlfo, i] = Qdown_d[iwlfo, i] + MQ * rho_hemi
        Qsig_d[:, i + 1] = Qsig_d[:, i] * wleaf * p
        Qsig_d[iwlfo, i + 1] = Qsig_d[iwlfo, i + 1] + MQ * p

    Qapar_bs = np.sum(Qapar + Qapar_d, axis=1)
    Qfdir_bs = np.sum(Qfdir + Qfdir_d, axis=1)
    Qfhemi_bs = np.sum(Qfhemi + Qfhemi_d, axis=1)
    Qfyld_bs = np.sum(Qfyld + Qfyld_d, axis=1)

    Qdown_bs = Qins * t0 + Qind * td + np.sum(Qdown + Qdown_d, axis=1)
    Qind_s = Qdown_bs * rs

    Qdown_bs_hot = Qins * t0
    Qind_s_hot = Qdown_bs_hot * rs

    Qdown_bs_d = Qind * td + np.sum(Qdown + Qdown_d, axis=1)
    Qind_s_d = Qdown_bs_d * rs

    Qfdir_s = np.zeros((nb, 11))
    Qfhemi_s = np.zeros((nb, 11))
    Qapar_s = np.zeros((nb, 11))
    Qdown_s = np.zeros((nb, 11))
    Qsig_s = np.zeros((nb, 12))
    Qfyld_s = np.zeros((nf, 11))

    Qapar_ss = np.zeros(nb)
    Qfdir_ss = np.zeros(nb)
    Qfhemi_ss = np.zeros(nb)
    Qfyld_ss = np.zeros(nf)

    for k in range(20):
        if k == 0:
            Qsig_s[:, 0] = Qind_s_hot * iD + Qind_s_d * iD
        else:
            Qsig_s[:, 0] = Qind_s * iD

        for i in range(11):
            Qapar_s[:, i] = Qsig_s[:, i] * (1 - wleaf)
            MQ = Mf @ Qsig_s[iwlfi, i]
            Qfdir_s[:, i] = Qsig_s[:, i] * wleaf * rho_obs
            Qfdir_s[iwlfo, i] = Qfdir_s[iwlfo, i] + MQ * rho_obs
            Qfhemi_s[:, i] = Qsig_s[:, i] * wleaf * rho_hemi
            Qfhemi_s[iwlfo, i] = Qfhemi_s[iwlfo, i] + MQ * rho_hemi
            Qfyld_s[:, i] = MQ
            Qdown_s[:, i] = Qsig_s[:, i] * wleaf * rho_hemi
            Qdown_s[iwlfo, i] = Qdown_s[iwlfo, i] + MQ * rho_hemi
            Qsig_s[:, i + 1] = Qsig_s[:, i] * wleaf * p
            Qsig_s[iwlfo, i + 1] = Qsig_s[iwlfo, i + 1] + MQ * p

        Qapar_ss += np.sum(Qapar_s, axis=1)

        if k == 0:
            Qfdir_ss += np.sum(Qfdir_s, axis=1) + Qins * rs * kg + Qind * rs * kg_dif + np.sum(Qdown + Qdown_d, axis=1) * rs * tv
            Qfhemi_ss += np.sum(Qfhemi_s, axis=1) + Qins * rs * kgd + Qind * rs * kgd_dif + np.sum(Qdown + Qdown_d, axis=1) * rs * td
        else:
            Qfdir_ss += np.sum(Qfdir_s, axis=1) + Qind_s * tv
            Qfhemi_ss += np.sum(Qfhemi_s, axis=1) + Qind_s * td

        Qfyld_ss += np.sum(Qfyld_s, axis=1)
        Qdown_ss = np.sum(Qdown_s, axis=1)
        Qind_s = Qdown_ss * rs

    Qfdir_all = Qfdir_bs + Qfdir_ss
    Qfhemi_all = Qfhemi_bs + Qfhemi_ss
    Qfyld_all = Qfyld_bs + Qfyld_ss
    Qapar_all = Qapar_bs + Qapar_ss

    return Qfdir_all, Qfhemi_all, Qfyld_all, Qapar_all


def cal_canopy_sif(d, netrad_sw_dict, M_pars, rtm_o_dict, hemi_dif_pars):
    rho_l, tau_l = d.leaf
    rho_l, tau_l = (rho_l[0:451, 0]).reshape(-1, 1), (tau_l[0:451, 0]).reshape(-1, 1)

    leaf = [rho_l, tau_l]
    soil = (d.soil[0:451]).reshape(-1, 1)

    [Mf2, Mb2, Mf1, Mb1] = M_pars

    Ma, Mb = np.zeros_like(Mf2), np.zeros_like(Mb2)

    d.MfII_diag[240:451, 0:351] = Mf2
    d.MbII_diag[240:451, 0:351] = Mb2

    d.MfI_diag[240:451, 0:351] = Mf1
    d.MbI_diag[240:451, 0:351] = Mb1

    d.MII_diag = d.wleaf_diag + d.MfII_diag + d.MbII_diag
    d.MI_diag = d.wleaf_diag + d.MfI_diag + d.MbI_diag
    d.MA_diag = d.wleaf_diag

    d.MII_diag_q = np.linalg.matrix_power(d.MII_diag, 2).astype(np.float32)
    d.MI_diag_q = np.linalg.matrix_power(d.MI_diag, 2).astype(np.float32)
    d.MA_diag_q = np.linalg.matrix_power(d.MA_diag, 2).astype(np.float32)

    p = rtm_o_dict['p']

    canopy_pars = [rtm_o_dict['i0'], rtm_o_dict['iD'], p, rtm_o_dict['rho2'], rtm_o_dict['rho_hemi'], rtm_o_dict['tv'],
                   rtm_o_dict['kc'], rtm_o_dict['kg']]
    dir_pars = [d.sob, d.sof, d.ko]
    hemi_pars = [rtm_o_dict['sob_vsla'], rtm_o_dict['sof_vsla'], rtm_o_dict['kgd']]
    dif_pars = [rtm_o_dict['sob_vsla_dif'], rtm_o_dict['sof_vsla_dif'], rtm_o_dict['kg_dif']]

    Qfdir_II, Qfhemi_II, Qfyld_II, _ = cal_rtm_sif(Mf2, Mb2, netrad_sw_dict, leaf, soil, canopy_pars, dir_pars,
                                                    hemi_pars, dif_pars, hemi_dif_pars)
    Qfdir_I, Qfhemi_I, Qfyld_I, _ = cal_rtm_sif(Mf1, Mb1, netrad_sw_dict, leaf, soil, canopy_pars, dir_pars, hemi_pars,
                                                 dif_pars, hemi_dif_pars)
    Qfdir_A, Qfhemi_A, Qfyld_A, _ = cal_rtm_sif(Ma, Mb, netrad_sw_dict, leaf, soil, canopy_pars, dir_pars, hemi_pars,
                                                 dif_pars, hemi_dif_pars)

    SRTE_Fs_fdir2 = Qfdir_II - Qfdir_A
    SRTE_Fs_fdir1 = Qfdir_I - Qfdir_A
    SRTE_Fs_fdir_all = SRTE_Fs_fdir1 + SRTE_Fs_fdir2

    SRTE_Fs_fhemi2 = Qfhemi_II - Qfhemi_A
    SRTE_Fs_fhemi1 = Qfhemi_I - Qfhemi_A
    SRTE_Fs_fhemi_all = SRTE_Fs_fhemi1 + SRTE_Fs_fhemi2

    SRTE_Fs_fyld2 = Qfyld_II
    SRTE_Fs_fyld1 = Qfyld_I
    SRTE_Fs_fyld_all = Qfyld_I + Qfyld_II

    nm_to_um = 1E-3
    SRTE_Fs_fdir2, SRTE_Fs_fdir1, SRTE_Fs_fdir_all = SRTE_Fs_fdir2 * nm_to_um, SRTE_Fs_fdir1 * nm_to_um, SRTE_Fs_fdir_all * nm_to_um
    SRTE_Fs_fhemi2, SRTE_Fs_fhemi1, SRTE_Fs_fhemi_all = SRTE_Fs_fhemi2 * nm_to_um, SRTE_Fs_fhemi1 * nm_to_um, SRTE_Fs_fhemi_all * nm_to_um

    return SRTE_Fs_fdir2, SRTE_Fs_fdir1, SRTE_Fs_fdir_all, SRTE_Fs_fhemi2, SRTE_Fs_fhemi1, SRTE_Fs_fhemi_all
