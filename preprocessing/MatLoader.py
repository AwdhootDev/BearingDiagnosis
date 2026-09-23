import scipy.io

def load_mat_file(uploaded_file):
    data = scipy.io.loadmat(uploaded_file)
    signal = None

    for key in data.keys():
        if key.endswith("_DE_time"):
            signal = data[key].flatten()
            break

    return signal
