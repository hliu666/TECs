import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from spectral_resampling import spectres

root = "C:/Users/liuha/Desktop/TBMm_DA/TBMm_DAv1/"
site_ID = 'US-xHA'
obs_df_h = pd.read_csv(root + f"flux/{site_ID}.csv")
obs_df_h = obs_df_h.dropna(subset=['406.99nm'])
obs_df_h = obs_df_h[(obs_df_h['month'] == 11)]

out_wl_ori = np.loadtxt(root + "band_info.txt")
out_wl = out_wl_ori[out_wl_ori <= 2100]
out_wl_fields = [f"{value}nm" for value in out_wl]

# Load the DataFrame
df = pd.read_csv("leaf_c14.csv")
df.columns = df.columns.str.replace(' ', '')
df = df.applymap(lambda x: x.replace(' ', '') if isinstance(x, str) else x)

# Filter out rows with 'Populus Tremuloides' in the 'spec_sci' field and 'BARK' in the 'speccode' field
sub_df = df[(df['plntpart'] == '"BARK"')]  # (df['spec_sci'] == 'Populus Tremuloides') &
sub_df = sub_df[['wavelen', 'reflect']]
sub_df.loc[:, 'wavelen'] = sub_df['wavelen'] * 1000.0
sub_df['reflect'] = pd.to_numeric(sub_df['reflect'], errors='coerce')

# Initialize variables to hold the spectra
spectra_list = []
current_spectrum = []

# Loop through the data to identify and extract each spectrum
prev_wavelen = sub_df['wavelen'].iloc[0]
for index, row in sub_df.iterrows():
    wavelen = row['wavelen']
    if wavelen < prev_wavelen and current_spectrum:
        spectra_list.append(pd.DataFrame(current_spectrum))
        current_spectrum = []
    current_spectrum.append(row)
    prev_wavelen = wavelen

out = []
for spectra in [spectra_list[4], spectra_list[8], spectra_list[11]]:
    # for spectra in spectra_list:
    spectra_sorted = spectra.sort_values(by='wavelen')
    spectra_sorted = spectra_sorted.drop_duplicates(subset='wavelen', keep='first')
    spectra_sorted = spectra_sorted[(spectra_sorted['wavelen'] >= 400) & (spectra_sorted['wavelen'] <= 2100)]
    new_fluxes = CubicSpline(spectra_sorted['wavelen'].values, spectra_sorted['reflect'].values)(out_wl)

    #plt.plot(out_wl, new_fluxes / max(new_fluxes), color='red')
    #plt.plot(out_wl, obs_df_h[out_wl_fields].values.flatten() / max(obs_df_h[out_wl_fields].values.flatten()), color='blue')
    #plt.show()

    out.append(new_fluxes)

out_arr = np.array(out)
mean_out_arr = np.nanmean(out_arr, axis=0)

# Load the DataFrame
df2 = pd.read_csv("Pine bark, Pinus sylvestris.csv")

# plt.plot(df2['#wlgth'], df2['mean'] / max(df2['mean']), color='green')
# plt.plot(out_wl, mean_out_arr / max(mean_out_arr), color='red')
# plt.plot(out_wl, obs_df_h[out_wl_fields].values.flatten() / max(obs_df_h[out_wl_fields].values.flatten()), color='blue')
plt.plot(df2['#wlgth'], df2['mean'], color='green')
plt.plot(out_wl, mean_out_arr, color='red')
plt.plot(out_wl, obs_df_h[out_wl_fields].values.flatten(), color='blue')
plt.show()
print("debug")
