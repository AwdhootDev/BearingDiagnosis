import numpy as np 
from scipy.stats import entropy

def extract_frequency_features(window):
    sampling_rate = 12000

    fft = np.fft.rfft(window)
    magnitude = np.abs(fft)

    frequencies = np.fft.rfftfreq(len(window), d=1/sampling_rate)

    dominant_frequency = frequencies[np.argmax(magnitude)]

    spectral_energy = np.sum(magnitude**2)

    spectral_centroid = np.sum(frequencies*magnitude / np.sum(magnitude))

    probablity = magnitude / np.sum(magnitude)
    spectral_entropy = entropy(probablity)

    spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid)**2)*magnitude) / np.sum(magnitude))

    peak_amplitude = np.max(magnitude)

    return {
        "Dominant_frequency" : dominant_frequency,
        "Spectral_energy" : spectral_energy,
        "Spectral_centroid" : spectral_centroid,
        "Spectral_entropy" : spectral_entropy,
        "Spectral_bandwidth" : spectral_bandwidth,
        "Peak_amplitude" : peak_amplitude
    }