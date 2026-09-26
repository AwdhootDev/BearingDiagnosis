import scipy.io

def load_mat_file(uploaded_file):
    data = scipy.io.loadmat(uploaded_file)
    signal = None

    for key in data.keys():
        if key.endswith("_DE_time"):
            signal = data[key].flatten()
            break

        elif "bearing" in key:
            mat_key = key
            signal = data[mat_key]
            signal = signal["gs"][0, 0].squeeze()
            break

    return signal
