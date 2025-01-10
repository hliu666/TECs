import sys

from forward_TBM_sub_stem.src.model.parameter import TBM_Pars
from forward_TBM_sub_stem.src.model.data import TBM_Data
from forward_TBM_sub_stem.src.model.model import TBM_Model
from post_unc.funcs.carbon_pool import carp_m, calc_nee
from post_unc.funcs.phe_func import calc_bud_onset

import pandas as pd
import numpy as np
import pickle


def run_tbm(root, site_ID, par_names, pars_arr):
    pars = pd.Series(pars_arr, index=par_names)

    rsr_red = np.genfromtxt("../model/support/rsr_red.txt")  #
    rsr_green = np.genfromtxt("../model/support/rsr_green.txt")  #
    rsr_blue = np.genfromtxt("../model/support/rsr_blue.txt")  #
    rsr_nir = np.genfromtxt("../model/support/rsr_nir1.txt")  #
    prospectpro = np.loadtxt("../model/support/dataSpec_PDB.txt")  #
    soil = np.genfromtxt("../model/support/soil_reflectance.txt")  #
    TOCirr = np.loadtxt("../model/support/atmo.txt", skiprows=1)  #
    # out_wl = np.loadtxt("../model/support/band_info.txt", encoding='ascii')  #
    out_wl = np.loadtxt("../../../band_info.txt", encoding='ascii')  #
    with open("../model/support/stem/data/stem.pkl", 'rb') as file:
        stem = pickle.load(file)

    mask = np.ones(out_wl.shape, dtype=bool)
    lower_bounds = np.array([1350, 1795, 2320])
    upper_bounds = np.array([1510, 2000, 2500])
    mask &= ~((out_wl[:, None] >= lower_bounds) & (out_wl[:, None] <= upper_bounds)).any(axis=1)

    filtered_out_wl = out_wl[mask]

    driving_data = [rsr_red, rsr_nir, rsr_green, rsr_blue, prospectpro, soil, TOCirr, out_wl, stem]

    hourly_obs = pd.read_csv(root + f"flux/{site_ID}.csv")
    daily_obs = pd.read_csv(root + f"flux_d/{site_ID}.csv")

    # hourly_obs = hourly_obs[(hourly_obs['year'] == 2020) | (hourly_obs['year'] == 2021)].reset_index(drop=True)
    # daily_obs = daily_obs[(daily_obs['year'] == 2020) | (daily_obs['year'] == 2021)].reset_index(drop=True)

    hourly_obs.loc[~hourly_obs['year'].isin([2017, 2018, 2019, 2020]), ['nee', 'nee_unc']] = np.nan
    hourly_obs.loc[~hourly_obs['year'].isin([2017, 2018, 2019, 2020]), filtered_out_wl] = np.nan
    daily_obs.loc[~daily_obs['year'].isin([2017, 2018, 2019, 2020]), ['lai', 'lai_std']] = np.nan

    hourly_obs, daily_obs = calc_bud_onset(hourly_obs, daily_obs)

    p = TBM_Pars(pars)
    d = TBM_Data(p, driving_data, hourly_obs)

    out_d_list = []
    for index_d, daily_row in daily_obs.iterrows():
        hourly_rows = hourly_obs[(hourly_obs['year'] == daily_row['year']) & (hourly_obs['doy'] == daily_row['doy'])]

        if index_d == 0:
            gpp_daily_2 = 0.001
            clab, cf, cr, cw, cl, cs = pars['clab'], pars['cf'], pars['cr'], pars['cw'], pars['cl'], pars['cs']
        else:
            gpp_daily_2 = gpp_daily
            clab, cf, cr, cw, cl, cs = clab2, cf2, cr2, cw2, cl2, cs2

        pars['d_onset'] = daily_row['sos'] + pars['del_onset'] + pars['cronset']
        carp_X = np.array([gpp_daily_2, daily_row['doy'], daily_row['ta'],
                           clab, cf, cr, cw, cl, cs,
                           pars['clspan'], pars['lma'], pars['f_auto'], pars['f_fol'],
                           pars['f_lab'], pars['f_roo'],
                           pars['Theta'], pars['theta_min'], pars['theta_woo'],
                           pars['theta_roo'], pars['theta_lit'], pars['theta_som'],
                           pars['d_onset'], pars['cronset'], pars['d_fall'],
                           pars['crfall'], pars['ci'],
                           ])

        if daily_row['doy'] < pars['d_onset'] or daily_row['doy'] > (pars['d_fall'] + pars['crfall']):
            stem_flag = True
            sai = pars['cw'] * pars['k_sai']
        elif daily_row['doy'] == pars['d_onset']:
            d.update_soil(ss_refl)
            stem_flag = False
            sai = None
        else:
            stem_flag = False
            sai = None

        lai, clab2, cf2, cr2, cw2, cl2, cs2, f_auto, Theta, theta_lit, theta_som = carp_m(carp_X)
        out_h_list, gpp_h_list = [], []
        for index_h, hourly_row in hourly_rows.iterrows():
            forcing = {
                'sw': hourly_row['sw'],
                'ta': hourly_row['ta'],
                'vpd': hourly_row['vpd'],
                'wds': hourly_row['wds'],
                'lai': lai.item(),
                'sza': hourly_row['sza_psm'],
                'vza': hourly_row['vza_psm'],
                'raa': hourly_row['raa_psm']
            }

            d.update(index_h, forcing)
            m = TBM_Model(d, p, stem_flag, sai)

            tbm_out_sub, ss_refl = m.tbm()
            nee, ra = calc_nee(tbm_out_sub[0], hourly_row['ta'], cl2, cs2, f_auto, Theta, theta_lit, theta_som)

            out_sub = np.concatenate(([hourly_row['year'], hourly_row['month'], hourly_row['day'], hourly_row['hour'],
                                       hourly_row['doy'], clab2, cf2, cr2, cw2, cl2, cs2],
                                      list(forcing.values()), pars, tbm_out_sub, [nee, ra]))

            out_h_list.append(out_sub)
            gpp_h_list.append(tbm_out_sub[0])

        out_h = np.array(out_h_list)
        gpp_daily = np.sum(gpp_h_list) * 1.03775 / 24

        out_d_list.append(out_h)

    out_d = np.array(out_d_list)
    out_d_re = out_d.reshape(-1, out_d.shape[2])

    out_df = pd.DataFrame(out_d_re,
                          columns=['year', 'month', 'day', 'hour', 'doy',
                                   'clab2', 'cf2', 'cr2', 'cw2', 'cl2', 'cs2',
                                   'sw', 'ta', 'vpd', 'wds', 'lai',
                                   'sza', 'vza', 'raa',
                                   "clab", "cf", "cr", "cw", "cl", "cs",
                                   "clspan", "lma", "f_auto", "f_fol", "f_lab", "f_roo",
                                   "Theta", "theta_min", "theta_woo", "theta_roo", "theta_lit", "theta_som",
                                   "del_onset", "cronset", "d_fall", "crfall",
                                   "Cab", "Vcmax25", "rsoil", "ci",
                                   "Car", "Cbrown", "Cw",
                                   "Rdsc", "BallBerrySlope", "BallBerry0", "k_sai", "d_onset",
                                   'ag', 'an', 'lst', 'fpar', 'brf_red', 'brf_nir'] +
                                  [f"{value}nm" for value in out_wl] +
                                  ['nee', 'ra'])

    out_df.to_csv('output_sim.csv', index=False)


if __name__ == '__main__':
    root = ""
    site_ID = 'US-xHA'

    # local optimization
    df = pd.read_csv(root + f"forward_TBM_sub_stem/src/opt/optimization_results.csv")
    pars = df.iloc[:, 1].values
    par_names = df.iloc[:, 0].values

    run_tbm(root, site_ID, par_names, pars)
