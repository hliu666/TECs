import numpy as np
from constants import O, P, T2K
from resistance_funcs import calc_d_0, calc_z_0M
from hydraulics_funcs import cal_thetas, hygroscopic_point, field_capacity, saturated_matrix_potential, calc_b

"""
Parameters for TBM model
"""


class TBM_Pars:
    def __init__(self, pars):
        """
        Canopy structure parameters
        Clumping Index (CI_flag):
            0: CI varied with the zenith angle
            1: CI as a constant 
            2: Without considering CI effect            
        """
        self.CI_thres = pars['ci']
        self.CI_flag = 1

        """
        Leaf Inclination Distribution Function
            * LIDF type     [a,b].
            * Planophile    [1,0].
            * Erectophile   [-1,0].
            * Plagiophile   [0,-1].
            * Extremophile  [0,1].
            * Spherical     [-0.35,-0.15].
            * Uniform       [0,0].
        """
        self.lidfa = -0.35
        self.lidfb = -0.15

        """
        Leaf traits parameters and constants 
        """
        self.ndub = 15

        self.Cab = pars['Cab']
        self.Car = pars['Car']
        self.Cm = pars['lma'] / 10000.0
        self.Cbrown = 0.2  # brown pigments concentration (unitless).
        self.Cw = 0.006  # equivalent water thickness (g cm-2 or cm).
        self.Ant = 1.96672  # Anthocianins concentration (mug cm-2).
        self.Alpha = 600  # constant for the optimal size of the leaf scattering element
        self.fLMA_k = 2765
        self.gLMA_k = 102.8
        self.gLMA_b = 0.0

        self.rho = 0.01  # [1]               Leaf/needle reflection
        self.tau = 0.01  # [1]               Leaf/needle transmission
        self.rs = 0.06  # [1]               Soil reflectance

        """
        Photosynthesis parameters and constants
        """
        # 1. Initial values of parameters
        self.kq = 300  # [umol e-1 umol sites-1 s-1] Cyt b6f kcat for PQH2
        self.Vcmax25 = pars['Vcmax25']  # [umol sites m-2] Rubisco density
        self.Rdsc = 0.015  # [] Scalar for mitochondrial (dark) respiration
        self.CB6F = 350/self.kq  # [umol sites m-2] Cyt b6f density
        self.gm = 0.15  # [] mesophyll conductance to CO2, ref: https://figshare.com/articles/dataset/A_global_dataset_of_mesophyll_conductance_measurements_and_accompanying_leaf_traits/19681410?file=36204663

        # 2. Initial values of constants
        self.BallBerrySlope = 8.0
        self.BallBerry0 = 0.01  # intercept of Ball-Berry stomatal conductance model
        self.Rd25 = self.Rdsc * self.Vcmax25

        # Cytochrome b6f-limited rates
        self.Kp1 = 14.5E9  # [s-1] Rate constant for photochemistry at PSI
        self.Kf = 0.05E9  # [s-1] Rate constant for fluoresence at PSII and PSI
        self.Kd = 0.55E9  # [s-1] Rate constant for constitutive heat loss at PSII and PSI

        # Note: a = a1 + a2 represents the total absorbed PAR
        self.a2 = 0.4420 / 0.85  # [] PSII, mol PPFD abs PS2 mol-1 PPFD incident
        self.a1 = 0.4080 / 0.85  # [] PSI, mol PPFD abs PS1 mol-1 PPFD incident

        self.nl = 0.75  # [ATP/e-] ATP per e- in linear flow
        self.nc = 1.00  # [ATP/e-] ATP per e- in cyclic flow

        self.spfy25 = 2444  # specificity (Computed from Bernacchhi et al. 2001 paper)
        self.ppm2bar = 1E-6 * (P * 1E-3)  # convert all to bar: CO2 was supplied in ppm, O2 in permil, and pressure in mBar
        self.O_c3 = (O * 1E-3) * (P * 1E-3)
        self.Gamma_star25 = 0.5 * self.O_c3 / self.spfy25  # [ppm] compensation point in absence of Rd

        # temperature correction for Kc
        self.Ec = 79430  # Unit is  [J K^-1]
        # temperature correction for Ko
        self.Eo = 36380  # Unit is  [J K^-1]
        # temperature correction for Gamma_star
        self.Eag = 37830  # Unit is [J K^-1]

        # temperature correction for Rd
        self.Ear = 46390  # Unit is [J K^-1]
        self.deltaSr = 490  # Unit is [J mol^-1 K^-1]
        self.Hdr = 150650  # Unit is [J mol^-1]

        self.Tref = 25 + T2K  # [K] absolute temperature at 25 degrees

        # temperature correction of Vcmax
        self.Eav = 65330  # Unit is [J K^-1]
        self.deltaSv = 485  # Unit is [J mol^-1 K^-1]
        self.Hdv = 149250  # Unit is [J mol^-1]

        self.Kc25 = 405 * 1E-6  # [mol mol-1]
        self.Ko25 = 279 * 1E-3  # [mol mol-1]

        self.minCi = 0.3

        """
        Fluorescence (Jen) parameters and constants 
        """
        # 1. Initial values of parameters
        self.fqe = 0.01

        # 1. Initial values of photochemical constants
        self.Kn1 = 14.5E9  # [s-1] Rate constant for regulated heat loss at PSI
        self.Kp2 = 4.5E9  # [s-1] Rate constant for photochemistry at PSII
        self.Ku2 = 0E9  # [s-1] Rate constant for exciton sharing at PSII

        self.eps1 = 0.0  # [mol PSI F to detector mol-1 PSI F emitted] PS I transfer function
        self.eps2 = 1.0  # [mol PSII F to detector mol-1 PSII F emitted] PS II transfer function

        """
        Resistence parameters and constants 
        """
        # 1. Initial values of parameters
        self.fwet = 0.0  # wetted fraction of the canopy
        self.leaf_width = 0.1  # efective leaf width size (m)
        self.h_C = 10.0  # vegetation height
        self.zm = 10.0  # Measurement height of meteorological data
        self.z_u = 10.0  # Height of measurement of windspeed (m).
        self.CM_a = 0.01  # Choudhury and Monteith 1988 leaf drag coefficient

        # 2. Initial values of constants
        self.d_0 = calc_d_0(self.h_C)  # displacement height
        self.z_0M = calc_z_0M(self.h_C)  # roughness length for momentum of the canopy

        """
        Phenology
        """
        self.radconv = 365.25 / np.pi

        """
        Soil parameters and constants 
        """
        self.sm0 = 0.6
        self.w0 = 0.0

        self.rsoil = pars['rsoil']  # brightness
        self.Soil = {
            "soc_top": [43, 39, 18],  # Soil composition, 1 by 3, percentages of sand, silt and clay
            "Zr_top": 0.5,
            "sti_top": 2,  # [] soil texture ID
        }

        self.Soil["theta_sat"] = cal_thetas(self.Soil['soc_top'])
        self.Soil["fc_top"] = field_capacity(self.Soil['soc_top'])
        self.Soil["sh_top"] = hygroscopic_point(self.Soil['soc_top'])

        self.Soil["phis_sat"] = saturated_matrix_potential(self.Soil["soc_top"][0])
        self.Soil["b1"] = calc_b(self.Soil["soc_top"][2])
