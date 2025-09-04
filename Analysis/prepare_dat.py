# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:19:47 2025

@author: Sianna.Groesser
"""

import pandas as pd
import os 
import numy as np

# --- function that checks if trigger in MEG data and in excel file for 
#       experiment match exactly ---

def verify_meg_trigs(mne_epochs, excel_col, excelfile_path, excel_filename): 
    
    '''
    takes mne.epochs object as input, reads in excel file with trials
     and then computes a match 
     
   Returns
   -------
   result : dict with keys:
       - shape_match : bool
       - order_match : bool
       - n_mismatches 
       - mismatch_table : pd.DataFrame (index, planned, presented)
       
      '''
    
    # load the excel file and extract the image triggers in the correct
    # order
    
    ## 
    excelfile = os.path.join(excelfile_path, excel_filename)
    df_planned_trials = pd.read_excel(excelfile)
    planned = df_planned_trials[excel_col].dropna().astype(int).to_numpy()
    
    presented = mne_epochs.events[:, 2].astype(int)
    
    
    # shape match
    n_planned = len(planned)
    n_presented = len(presented)
    shape_match = (n_planned == n_presented)

    
    # content match 
    n_compare = min(n_planned, n_presented)
    mism_idx = np.where(planned[:n_compare] != presented[:n_compare])[0]
    order_match = (len(mism_idx) == 0) 
    
    mismatch_table = pd.DataFrame({
       "index": mism_idx,
       "planned": planned[mism_idx] if len(mism_idx) else np.array([], dtype=int),
       "presented": presented[mism_idx] if len(mism_idx) else np.array([], dtype=int),
   })

   result = {
       "shape_match": shape_match,
       "order_match": order_match,
       "n_mismatches": len(mism_idx),
       "mismatch_table": mismatch_table,
   }
   return result