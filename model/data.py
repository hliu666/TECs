import numpy as np

from RTM_initial import sip_leaf, soil_spectra, atmoE
from RTM_initial import cal_lidf, weighted_sum_over_lidf_vec, CIxy
from RTM_initial import hemi_initial, dif_initial, hemi_dif_initial
from Ebal_funcs import calc_vapor_pressure
from RTM_initial import fluspect

xrange = range


class TBM_Data:
    """
    Data for TBM model
    """

    def __init__(self, p, data, obs):
        """ Extracts data from netcdf file
        :param start_yr: year for model runs to begin as an integer (year)
        :param end_yr: year for model runs to end as an integer (year)
        :return:
        """

        [rsr_red, rsr_nir, rsr_green, rsr_blue, prospectpro, soil, TOCirr, out_wl, stem] = data

        # 'Driving Data'
        self.sw = None
        self.t_mean = None
        self.ea = None
        self.wds = None
        self.lai = None

        self.Cab = p.Cab
        self.Car = p.Car
        self.Cm = p.Cm
        self.Cbrown = p.Cbrown  # brown pigments concentration (unitless).
        self.Cw = p.Cw  # equivalent water thickness (g cm-2 or cm).
        self.Ant = p.Ant  # Anthocianins concentration (mug cm-2).
        self.Alpha = p.Alpha  # constant for the optimal size of the leaf scattering element
        self.fLMA_k = p.fLMA_k
        self.gLMA_k = p.gLMA_k
        self.gLMA_b = p.gLMA_b

        """ 
        Initialization of Leaf-level SIF  
        """
        self.leaf = sip_leaf(prospectpro, self.Cab, self.Car, self.Cbrown, self.Cw, \
                             self.Cm, self.Ant, self.Alpha, self.fLMA_k, self.gLMA_k, \
                             self.gLMA_b, p.tau, p.rho)

        stem_rho = np.vstack((stem[0].reshape(-1, 1), np.full((2162 - 2100, 1), p.rho)))
        stem_tau = np.vstack((stem[1].reshape(-1, 1), np.full((2162 - 2100, 1), p.tau)))
        self.stem = [stem_rho, stem_tau]

        """
        self.fluspect_dict = fluspect(p.ndub, optipar, prospectpro, p.Cab, p.Car, p.Cbrown, p.Cw, p.Cm, p.Ant, p.Alpha)

        # Initialization of Canopy-level SIF  
        self.wleaf = self.leaf[0] + self.leaf[1]
        self.wleaf_diag = np.diag(self.wleaf[0:451].flatten()).astype(np.float32)
        self.aleaf_diag = np.diag(1 - self.wleaf[0:451].flatten()).astype(np.float32)

        self.MbI_diag = np.zeros_like(self.wleaf_diag).astype(np.float32)
        self.MfI_diag = np.zeros_like(self.wleaf_diag).astype(np.float32)

        self.MbII_diag = np.zeros_like(self.wleaf_diag).astype(np.float32)
        self.MfII_diag = np.zeros_like(self.wleaf_diag).astype(np.float32)

        self.MbA_diag = np.zeros_like(self.wleaf_diag).astype(np.float32)
        self.MfA_diag = np.zeros_like(self.wleaf_diag).astype(np.float32)

        self.MI_diag = np.zeros_like(self.wleaf_diag).astype(np.float32)
        self.MII_diag = np.zeros_like(self.wleaf_diag).astype(np.float32)
        """

        """ 
        Initialization of soil model
        """
        self.soil = soil_spectra(soil, p.rsoil, p.rs)

        """
        The spectral response curve 
        """
        self.rsr_red = rsr_red
        self.rsr_nir = rsr_nir
        self.rsr_green = rsr_green
        self.rsr_blue = rsr_blue

        self.sim_wl = np.arange(400, 2500)
        self.out_wl = out_wl

        """ 
        Initialization of sun's spectral curve
        """
        self.wl, self.atmoMs = atmoE(TOCirr)

        """
        Sun-sensor geometry
        """
        self.tts = obs['sza_psm'].values
        self.tto = obs['vza_psm'].values
        self.psi = obs['raa_psm'].values

        """
        Initialization of leaf angle distribution
        """
        self.lidf = cal_lidf(p.lidfa, p.lidfb)

        """
        Clumping Index (CI_flag)      
        """
        self.CIs = p.CI_thres  # CIxy(p.CI_flag, self.tts, p.CI_thres)
        self.CIo = p.CI_thres  # CIxy(p.CI_flag, self.tto, p.CI_thres)

        """ 
        Initialization of canopy-level Radiative Transfer Model 
        """
        self.ks_all, self.ko_all, _, self.sob_all, self.sof_all = weighted_sum_over_lidf_vec(self.lidf, self.tts, self.tto, self.psi)

        self.hemi_pars_all = hemi_initial(self.tts, self.lidf)
        self.dif_pars_all = dif_initial(self.tto, self.lidf)
        self.hemi_dif_pars = hemi_dif_initial(self.lidf)

        self.ks = None
        self.ko = None
        self.sob = None
        self.sof = None

        self.hemi_pars = None
        self.dif_pars = None

        """
        Initialization of hydraulics model
        """
        self.sm_top = p.sm0
        self.w_can = p.w0

    def update(self, i, forcing):
        self.sw = forcing['sw']
        self.t_mean = forcing['ta']
        self.ea = calc_vapor_pressure(self.t_mean) - forcing['vpd']
        self.wds = forcing['wds']
        self.lai = max(forcing['lai'], 1e-16)

        """
        Sun-sensor geometry
        """
        self.tts = np.array([forcing['sza']])
        self.tto = np.array([forcing['vza']])
        self.psi = np.array([forcing['raa']])

        self.ks, self.ko, self.sob, self.sof = np.array([self.ks_all[i]]), np.array([self.ko_all[i]]), np.array([self.sob_all[i]]), np.array([self.sof_all[i]])
        self.hemi_pars = [arr[i*64:(i+1)*64] for arr in self.hemi_pars_all]
        self.dif_pars = [arr[i*64:(i+1)*64] for arr in self.dif_pars_all]

    def update_soil(self, stem_refl):
        self.soil = stem_refl
