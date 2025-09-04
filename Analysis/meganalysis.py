# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 16:20:03 2025

@author: Sianna.Groesser
"""

import numpy as np
import warnings 

from scipy.spatial.distance import pdist, squareform

def get_channels(epochs, picks):
    """
    Get channel names and indices from an MNE Epochs object by name or by type.
    
    Parameters
    ----------
    epochs : mne.Epochs
    picks : str | list of str
        Channel names (e.g. 'MEG121') and/or channel types (e.g. 'grad', 'mag', 'eeg').

    Returns
    -------
    ch_names : list of str
        Channel names that match the picks.
    ch_indices : list of int
        Indices of those channels in epochs.ch_names.
    """
    if isinstance(picks, str):
        picks = [picks]

    all_ch_names = epochs.ch_names
    all_ch_types = epochs.get_channel_types()

    ch_names, ch_indices = [], []
    for idx, (ch_name, ch_type) in enumerate(zip(all_ch_names, all_ch_types)):
        if (ch_name in picks) or (ch_type in picks):
            ch_names.append(ch_name)
            ch_indices.append(idx)

    return ch_names, ch_indices

def compute_rsa_meg(input_data, labels):
    
    """
    computes representational dissimilarity matrices (RDMs) for MEG data 
    
    Parameters
    ----------
    input_data : 2d-array, needs to have shape n_trials * n_sensors (either
        average over time points first select specific time)
    labels : array with condition labels, shape n_trials
    """
    
    
    # seond step: average over trials from the same condition if necessary
    # check that there is same number of trials for all conditions
    unique_labels, counts = np.unique(labels, return_counts=True)
    if not np.all(counts == counts[0]):
        warnings.warn(f"Unequal counts: {dict(zip(unique_labels, counts))}")
    
    n_labels = len(unique_labels)
    n_sensors = np.shape(input_data)[1]
    per_label_means = np.zeros((n_labels, n_sensors))
    
    for idx, lab in enumerate(unique_labels):
        label_mask = labels==lab
        data = input_data[label_mask,:]
        per_label_means[idx] = data.mean(axis=0)
   
    
    # third step: take pairs of conditions and correlate
    rdm_corr = squareform(pdist(per_label_means, metric="correlation"))
    
    return rdm_corr, per_label_means