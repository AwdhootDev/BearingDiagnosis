import scipy.io

def load_mat_file(uploaded_file):
    data = scipy.io.loadmat(uploaded_file)
    signal = None
    sr = 12000

    for key in data.keys():
        if key.endswith("_DE_time"):
            signal = data[key].flatten()
            break

        elif "bearing" in key:
            mat_key = key
            signal = data[mat_key]["gs"][0, 0].squeeze()
            try:
                sr = int(data[mat_key]["sr"][0, 0].flatten()[0])
            except:
                pass
            break

    return signal, sr
