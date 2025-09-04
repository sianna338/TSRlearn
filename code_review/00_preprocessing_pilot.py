# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: env-tsr-learn
#     language: python
#     name: python3
# ---

# %%
import mne
import numpy as np
import os
import fnmatch
import pandas as pd
import sklearn
import sigproc
import decoding
import openpyxl
from matplotlib import pyplot as plt


# %%
def plotting_function(input_data): 

    """ plot data in channel x time format """

    plt.matshow(input_data)    


# %% [markdown]
# Analysis Settings

# %%
data_filepath = r"c:\\sync_folder\\TSRlearn\\Data\\"
os.chdir(data_filepath)
subject_name = "pilot_sianna"

data_path = os.path.join(data_filepath + subject_name)

tint_epoching = (-0.1, 0.5) # time_interval for segmenting data around image presentation 


# %% [markdown]
# Load raw data

# %%
all_files = os.listdir(data_path)

fif_files = fnmatch.filter(all_files, '*tsss*.fif')
data_raw = mne.io.read_raw_fif((data_path + '//' + fif_files[1]), verbose=False)
data_raw

# %% [markdown]
# Plot raw data

# %%
data_raw.plot(
   n_channels=15,          
        duration=120.0,  
        start=1000,        
        scalings= dict(mag=4e-12, grad=40e-12, eeg=150e-6),
        color=dict(mag='navy', grad='purple')
)

# %% [markdown]
# Find triggers in the dataset and map them to event names 

# %%
# events
events = mne.find_events(data_raw, 
                'STI101', min_duration=0.005)

# create df with information 
df_events = pd.DataFrame(events, columns=['sample', 'prev_value', 'event_id'])
print(df_events)
# Convert samples to time (in hours)
sfreq = data_raw.info['sfreq']
df_events['time_sec'] = (df_events['sample'] / sfreq) 

# Optional: Map trigger codes to meaningful names
event_dict = {
    255: 'Start / End Experiment',
    99: 'Word Presented',
    127: 'Response',
    97: 'Match Prompt Response',
    96: 'Feedback',
    95: 'too slow - message',
    94: 'Fixation Cross'
}

# Add triggers 1–21 all mapped to "Image Presentation"
event_dict.update(dict.fromkeys(range(1, 22), "Image Presented"))

df_events['event_name'] = df_events['event_id'].map(event_dict).fillna('Unknown')

# Reorder columns for clarity
df_events = df_events[['time_sec', 'sample', 'event_id', 'event_name']]

# --- Display the first few rows ---
print(df_events.head(10))

# --- Save full table ---
df_events.to_excel("pilot1_all-triggers-read.xlsx", index=False)

# %% [markdown]
# Resample 

# %%
# resample 
# data = data_raw.load_data(dtype = 'float32')
# data_resampled = sigproc.resample(data_raw, 1000,100)
data_raw.resample(100)

# %% [markdown]
# Plot resampled data

# %%
# plot for sanity check
data_raw.plot(n_channels=15,          
        duration=120.0,  
        start=1000,        
        scalings= dict(mag=4e-12, grad=40e-12, eeg=150e-6),
        color=dict(mag='navy', grad='purple'))

# %% [markdown]
# Bandpass Filter

# %%
# BP filter (0.5 - 40 Hz)
data_raw.filter(l_freq=0.5, h_freq=40) 

# %% [markdown]
# Segment around times of image presentation

# %% [markdown]
# Plot filtered data

# %%
# plot for sanity check
data_raw.plot(n_channels=15,          
        duration=120.0,  
        start=1000,        
        scalings= dict(mag=4e-12, grad=40e-12, eeg=150e-6),
        color=dict(mag='navy', grad='purple'))

# %% [markdown]
# Segment data

# %%
# filter only event trigger where images where presented

events = mne.find_events(data_raw, stim_channel='STI101', shortest_event=1)
events_img_pres = events[np.isin(events[:, 2], np.arange(1, 22))]


print("number of img presentation events: ", len(events_img_pres))
print(events_img_pres)

# %%

# option 1: numpy
unique, counts = np.unique(events_img_pres[:,2], return_counts=True)
for u, c in zip(unique, counts):
    print(f"Event code {u}: {c} times")

# (optional) restrict to certain IDs, e.g. 1..21
sel = np.isin(events_img_pres[:, 2], np.arange(1, 22))
ev = events_img_pres[sel]

sfreq = data_raw.info['sfreq']

# duration (sec) until next event; last gets inf so it won't be the minimum
dur_sec = np.diff(ev[:, 0], append=np.inf) / sfreq

# print all durations
for i, d in enumerate(dur_sec):
    print(f"Event #{i}: code={ev[i, 2]} duration≈{d:.4f}s")

# index of the shortest (ignore the last inf automatically)
idx_short = int(np.argmin(dur_sec[:-1]))

print(f"\nShortest event: index={idx_short}, code={ev[idx_short, 2]}, dur≈{dur_sec[idx_short]:.4f}s")

# exclude the shortest from your events array
ev_wo_short = np.delete(ev, idx_short, axis=0)

# %%
# segment around times of imahge presentation 
data_epoched_img = mne.Epochs(data_raw, ev_wo_short, tmin=tint_epoching[0], tmax=tint_epoching[1],preload=True, baseline=None)


# %% [markdown]
# Save preprocessed data

# %%
# safe as .fif. file to use later
# data_preprocessed = data_raw 
data_epoched_img.save(os.path.join(data_filepath , subject_name, "pilot1_Sianna_epoched_raw.fif"), overwrite=True)
