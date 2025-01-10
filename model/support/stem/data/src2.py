# ref: https://www.aai.ee/bgf/ger2600/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from spectral_resampling import spectres
import pickle
import csv


def moving_average(data, window_size):
    extended_data = np.pad(data, (window_size//2, window_size//2), mode='edge')
    smoothed_data = np.convolve(extended_data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data


df2 = pd.read_csv("Pine bark, Pinus sylvestris.csv")
df2 = df2[['#wlgth', 'mean', 'stdev']]
df2['mean'] = df2['mean'] * 1.25

refl = interp1d(df2['#wlgth'].values, df2['mean'].values, kind="slinear", fill_value="extrapolate")(np.arange(400, 2500))
tran = np.full_like(refl, 0.001)

window_size = 5
smoothed_refl = moving_average(refl, window_size)

list = [smoothed_refl, tran]
# Save the list to a .pkl file
with open("stem.pkl", 'wb') as file:
    pickle.dump(list, file)

with open('stem.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(list)


# smoothed_refl[1350:1510] = np.nan
# smoothed_refl[1795:2000] = np.nan
# smoothed_refl[2320:2500] = np.nan

# plt.plot(df2['#wlgth'].values, df2['mean'].values, color='red')
# plt.plot(smoothed_refl, color='blue')
# lt.show()
