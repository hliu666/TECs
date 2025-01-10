# ==============================================================================
# List of constants used in Meteorological computations
# ==============================================================================
# convert temperatures to K
T2K = 273.15
# Stephan Boltzmann constant (W m-2 K-4)
sigmaSB = 5.670373e-8

# heat capacity of dry air at constant pressure (J kg-1 K-1)
c_pd = 1003.5
# heat capacity of water vapour at constant pressure (J kg-1 K-1)
c_pv = 1865
# ratio of the molecular weight of water vapor to dry air
epsilon = 0.622
# Psicrometric Constant kPa K-1
psicr = 0.0658
# gas constant for dry air, J/(kg*degK)
R_d = 287.04
# von Karman's constant
KARMAN = 0.41
# acceleration of gravity (m s-2)
GRAVITY = 9.8
# Molecular mass of water [g mol-1]
MH2O = 18
# Molecular mass of dry air [g mol-1]
Mair = 28.96
# Air pressure
P = 970
# leaf water
eb = 27.0
# Atmospheric O2 concentration
O = 209
# atmospheric CO2 concentration
Ca = 390
# leaf CO2 concentration
Cs = 250
# Conversion of vapour pressure [Pa] to absolute humidity [kg kg-1]
e_to_q = MH2O / Mair / P
# Ideal gas constant
R = 8.314  # [J mol-1 K-1]

U_FRICTION_MIN = 0.01

U_C_MIN = 0.01

# ==============================================================================
# List of constants used in Atmospheric longwave radiation
# ==============================================================================
# heat capacity of dry air at constant pressure (J kg-1 K-1)
c_pd = 1003.5
# heat capacity of water vapour at constant pressure (J kg-1 K-1)
c_pv = 1865
# ratio of the molecular weight of water vapor to dry air
epsilon = 0.622
# gas constant for dry air, J/(kg*degK)
R_d = 287.04

Rhoa = 1.2047  # [kg m-3]      Specific mass of air
Mair = 28.96  # [g mol-1]     Molecular mass of dry air
RGAS = 8.314  # [J mol-1K-1]   Molar gas constant
