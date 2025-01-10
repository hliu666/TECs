import joblib
import pandas as pd
import numpy as np
import scipy.io
import sys
import mat4py

from data import TBM_Data
from parameter import TBM_Pars
from model import TBM_Model


def main():
    rsr_red = np.genfromtxt("support/rsr_red.txt")  #
    rsr_green = np.genfromtxt("support/rsr_green.txt")  #
    rsr_blue = np.genfromtxt("support/rsr_blue.txt")  #
    rsr_nir = np.genfromtxt("support/rsr_nir1.txt")  #
    prospectpro = np.loadtxt("support/dataSpec_PDB.txt")  #
    soil = np.genfromtxt("support/soil_reflectance.txt")  #
    TOCirr = np.loadtxt("support/atmo.txt", skiprows=1)  #
    optipar = mat4py.loadmat("../model/support/Optipar_ProspectPRO_CX.mat")['optipar']
    out_wl = np.loadtxt("support/band_info.txt", encoding='ascii')  #

    driving_data = [rsr_red, rsr_nir, rsr_green, rsr_blue, prospectpro, soil, TOCirr, optipar, out_wl]

    # i, js = int(sys.argv[11]), int(sys.argv[12])
    i, js = 0, 6650

    for j in range(0, js):
        k = i + j

        # inputs = np.load(sys.argv[10])[k]
        inputs = np.load("../../data/parameters/input_data_test.npy")[k]
        field_names = ['sw', 'ta', 'vpd', 'wds', 'lai', 'sza', "vza", "raa", "cab", "lma", "rsoil", 'Vcmax25']
        inputs_dict = dict(zip(field_names, inputs))

        p = TBM_Pars(inputs_dict)
        d = TBM_Data(p, inputs_dict, driving_data)
        m = TBM_Model(d, p)

        tbm_out_sub = m.tbm()
        out_sub = np.concatenate((inputs, tbm_out_sub))

        if j == 0:
            out = out_sub
        else:
            out = np.vstack((out, out_sub))

    out_df = pd.DataFrame(out,
                          columns=['sw', 'ta', 'vpd', 'wds', 'lai',
                                   'sza', 'vza', 'raa',
                                   "cab", "lma", "rsoil", "Vcmax25",
                                   'ag', 'an', 'lst', 'fpar', 'brf_red', 'brf_nir'] +
                                  [f"{value}nm" for value in out_wl])

    out_df.to_csv('{0}_forward_hourly.csv'.format(i // js), index=False)


if __name__ == '__main__':
    main()
