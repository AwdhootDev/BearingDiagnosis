import numpy as np
from scipy.stats import skew, kurtosis

def extract_features(window):
    mean = np.mean(window)
    std = np.std(window)
    maximum = np.max(np.abs(window))

    rms = np.sqrt(np.mean(window**2))
    kurt = kurtosis(window)
    skewness = skew(window)
    p2p = np.ptp(window)

    crest_factor = maximum/rms
    mean_abs = np.mean(np.abs(window))
    shape_factor = rms/mean_abs
    impulse_factor = maximum/mean_abs
    clearance_factor = maximum/(np.mean(np.sqrt(np.abs(window)))**2)

    return {
        'Mean' : mean,
        'Std' : std,
        'RMS' : rms, 
        'Kurtosis': kurt, 
        'Peak_to_Peak' : p2p, 
        'Skewness' : skewness,
        'Crest_Factor' : crest_factor,
        'Shape_Factor' : shape_factor,
        'Impulse_Factor' : impulse_factor,
        'Clearance_Factor' : clearance_factor,
    }