import numpy as np
import mne
from hypyp.prep import prep_meeg
from hypyp.analyses import compute_sync

def epoch_continuous_data(raw_p1, raw_p2, epoch_length=1.0):
    sfreq = raw_p1.info['sfreq']
    
    events_p1 = mne.make_fixed_length_events(raw_p1, duration=epoch_length)
    events_p2 = mne.make_fixed_length_events(raw_p2, duration=epoch_length)
    
    epochs_p1 = mne.Epochs(raw_p1, events_p1, tmin=0, tmax=epoch_length, baseline=None, preload=True, verbose=False)
    epochs_p2 = mne.Epochs(raw_p2, events_p2, tmin=0, tmax=epoch_length, baseline=None, preload=True, verbose=False)
    
    prepped_data, _ = prep_meeg(epochs_p1, epochs_p2, verbose=False)
    
    return prepped_data, sfreq

def compute_psi_plv(prepped_data, sfreq, freq_band):
    
    # MATLABコードと同等（Morletウェーブレット変換 + PLV算出）の処理を行う。
    results = compute_sync(
        data=prepped_data,
        mode='CC',
        metric='plv',          
        frequencies=freq_band,
        sampling_rate=sfreq,
        verbose=False
    )
    

    psi_matrix = results
    
    return psi_matrix