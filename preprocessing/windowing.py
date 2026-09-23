import numpy as np

def create_windows(signal, window_size=2048, step=None):
    signal = np.asarray(signal).flatten()

    if step is None:
        step = window_size

    windows = []

    for start in range(0, len(signal) - window_size, step):
        window = signal[start:start+window_size]
        windows.append(window)

    return np.array(windows)
