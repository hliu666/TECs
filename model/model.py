"""
TBM model class takes a data class and then uses functions to run the TBM model.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from constants import T2K

from RTM_Optical import rtm_o, BRF_hemi_dif_func
from Ebal import Ebal
from PhotoSynth_Jen import PhotoSynth_Jen, peaked_arrh
from SIF import cal_sif_leaf, cal_canopy_sif

xrange = range


class TBM_Model:

    def __init__(self, dataclass, pramclass, stem_flag, sai):
        """ Model class for running DALEC2
        :param dataclass: TBM data class containing data to run model
        :return:
        """
        self.d = dataclass
        self.p = pramclass
        self.stem_flag = stem_flag
        self.sai = sai

    # ------------------------------------------------------------------------------
    # Model functions
    # ------------------------------------------------------------------------------
    def tbm(self):

        lai = self.d.lai

        hemi_dif_pars = BRF_hemi_dif_func(self.d.hemi_dif_pars, self.d.CIs, self.d.CIo, lai)
        if self.stem_flag:
            rtm_stem_pars = rtm_o(self.d, self.p, self.sai, hemi_dif_pars, self.stem_flag)
        rtm_pars = rtm_o(self.d, self.p, lai, hemi_dif_pars, False)
        ebal_pars, netrad_sw_pars = Ebal(self.d, self.p, lai, rtm_pars)

        if (self.d.tts < 75) and \
                (netrad_sw_pars['ERnuc'] > 1) and \
                (netrad_sw_pars['ERnhc'] > 1) and \
                (lai > 0.1):
            # ----------------------canopy intercepted wator and soil moisture factor---------------------

            # ----------------------two leaf model---------------------
            APARu_leaf = netrad_sw_pars['APARu'] / (lai * netrad_sw_pars['Fc'])
            APARh_leaf = netrad_sw_pars['APARh'] / (lai * (1 - netrad_sw_pars['Fc']))

            meteo_u = [APARu_leaf, ebal_pars['Ccu'], ebal_pars['Tcu'], ebal_pars['ecu']]
            meteo_h = [APARh_leaf, ebal_pars['Cch'], ebal_pars['Tch'], ebal_pars['ech']]

            rcw_u, _, Agu, Anu, fqe2u, fqe1u = PhotoSynth_Jen(meteo_u, self.p)
            rcw_h, _, Agh, Anh, fqe2h, fqe1h = PhotoSynth_Jen(meteo_h, self.p)

            Ag = (Agu * netrad_sw_pars['Fc'] + Agh * (1 - netrad_sw_pars['Fc'])) * lai
            An = (Anu * netrad_sw_pars['Fc'] + Anh * (1 - netrad_sw_pars['Fc'])) * lai

            eta_u_pars = [fqe2u, fqe1u]
            eta_h_pars = [fqe2h, fqe1h]

            """
            Mu_pars = cal_sif_leaf(self.p, self.d.fluspect_dict, eta_u_pars)
            Mh_pars = cal_sif_leaf(self.p, self.d.fluspect_dict, eta_h_pars)

            sif_dir2_u, sif_dir1_u, sif_dir_all_u, sif_hemi2_u, sif_hemi1_u, sif_hemi_all_u \
                = cal_canopy_sif(self.d, netrad_sw_pars, Mu_pars, rtm_pars, hemi_dif_brf)
            sif_dir2_h, sif_dir1_h, sif_dir_all_h, sif_hemi2_h, sif_hemi1_h, sif_hemi_all_h \
                = cal_canopy_sif(self.d, netrad_sw_pars, Mh_pars, rtm_pars, hemi_dif_brf)
            """
        else:
            Rdu = -peaked_arrh(self.p.Rd25, self.p.Ear, ebal_pars['Tcu'] + T2K, self.p.deltaSr, self.p.Hdr)
            Rdh = -peaked_arrh(self.p.Rd25, self.p.Ear, ebal_pars['Tch'] + T2K, self.p.deltaSr, self.p.Hdr)

            Ag = np.array([0])
            An = (Rdu * netrad_sw_pars['Fc'] + Rdh * (1 - netrad_sw_pars['Fc'])) * lai

            """
            sif_dir2_u, sif_dir1_u, sif_dir_all_u, sif_hemi2_u, sif_hemi1_u, sif_hemi_all_u \
                = np.zeros(451), np.zeros(451), np.zeros(451), np.zeros(451), np.zeros(451), np.zeros(451)
            sif_dir2_h, sif_dir1_h, sif_dir_all_h, sif_hemi2_h, sif_hemi1_h, sif_hemi_all_h \
                = np.zeros(451), np.zeros(451), np.zeros(451), np.zeros(451), np.zeros(451), np.zeros(451)

            fqe2u, fqe1u, fqe2h, fqe1h = np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
            """

        sur_refl = rtm_pars['BRF'] * netrad_sw_pars['ratio'] + rtm_pars['BRF_dif'] * (1 - netrad_sw_pars['ratio'])
        sur_refl_red = float(np.nansum(sur_refl[220:271].flatten() * self.d.rsr_red.flatten()) / np.nansum(self.d.rsr_red.flatten()))
        sur_refl_nir = float(np.nansum(sur_refl[441:477].flatten() * self.d.rsr_nir.flatten()) / np.nansum(self.d.rsr_nir.flatten()))

        brf_refl = rtm_pars['BRF']
        brf_refl_red = float(np.nansum(brf_refl[220:271].flatten() * self.d.rsr_red.flatten()) / np.nansum(self.d.rsr_red.flatten()))
        brf_refl_nir = float(np.nansum(brf_refl[441:477].flatten() * self.d.rsr_nir.flatten()) / np.nansum(self.d.rsr_nir.flatten()))

        brf_refl_spectrum = brf_refl[0:2100].flatten()
        brf_refl_spline = CubicSpline(self.d.sim_wl, brf_refl_spectrum)
        resample_brf_refl_spectrum = brf_refl_spline(self.d.out_wl)

        # fpar = sum((rtm_o_dict['A'] * netrad_sw_dict['ratio'] + rtm_o_dict['A_dif'] * (1 - netrad_sw_dict['ratio']))[0:301]) / 301
        """
        sif_dir2_u, sif_dir1_u, sif_dir_all_u, sif_hemi2_u, sif_hemi1_u, sif_hemi_all_u = (
            sif_dir2_u[250:400], sif_dir1_u[250:400], sif_dir_all_u[250:400],
            sif_hemi2_u[250:400], sif_hemi1_u[250:400], sif_hemi_all_u[250:400])

        sif_dir2_h, sif_dir1_h, sif_dir_all_h, sif_hemi2_h, sif_hemi1_h, sif_hemi_all_h = (
            sif_dir2_h[250:400], sif_dir1_h[250:400], sif_dir_all_h[250:400],
            sif_hemi2_h[250:400], sif_hemi1_h[250:400], sif_hemi_all_h[250:400])

        sif = np.hstack((sif_dir2_u, sif_dir1_u, sif_dir_all_u, sif_hemi2_u, sif_hemi1_u, sif_hemi_all_u,
                         sif_dir2_h, sif_dir1_h, sif_dir_all_h, sif_hemi2_h, sif_hemi1_h, sif_hemi_all_h))

        sif_dir = np.nanmean((sif_dir_all_u + sif_dir_all_h)[93:109])

        sif_dir2_u = np.nanmean(sif_dir2_u[93:109])
        sif_dir1_u = np.nanmean(sif_dir1_u[93:109])
        sif_dir2_h = np.nanmean(sif_dir2_h[93:109])
        sif_dir1_h = np.nanmean(sif_dir1_h[93:109])
        """

        if self.stem_flag:
            brf_refl = rtm_stem_pars['BRF']
            brf_refl_red = float(np.nansum(brf_refl[220:271].flatten() * self.d.rsr_red.flatten()) / np.nansum(self.d.rsr_red.flatten()))
            brf_refl_nir = float(np.nansum(brf_refl[441:477].flatten() * self.d.rsr_nir.flatten()) / np.nansum(self.d.rsr_nir.flatten()))

            brf_refl_spectrum = brf_refl[0:2100].flatten()
            brf_refl_spline = CubicSpline(self.d.sim_wl, brf_refl_spectrum)
            resample_brf_refl_spectrum = brf_refl_spline(self.d.out_wl)

        out = np.concatenate(([Ag[0], An[0], ebal_pars['LST'][0], netrad_sw_pars['fPAR'][0], brf_refl_red, brf_refl_nir], resample_brf_refl_spectrum))

        if self.stem_flag:
            ss_refl = rtm_stem_pars['R_dif']
        else:
            ss_refl = None

        return out, ss_refl
