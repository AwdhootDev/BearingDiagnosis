import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.MatLoader import load_mat_file
from preprocessing.windowing import create_windows
from preprocessing.time_features import extract_features
from preprocessing.frequency_features import extract_frequency_features

st.title("Bearing Fault Diagnosis")
st.write("Upload a vibration signal .mat file for analysis")

uploaded_file = st.file_uploader("Upload MATLAB (.mat) file", type=["mat"])

if uploaded_file is not None:

    signal = load_mat_file(uploaded_file)

    st.write(f"Signal shape : {signal.shape}")

    st.success(f"File uploaded: {uploaded_file.name}")

    st.session_state["signal"] = signal
    st.success("Signal extracted successfully!!")

    if "signal" in st.session_state:
        windows_2048 = create_windows(signal, window_size=2048)

        if windows_2048 is not None:
            fig, ax = plt.subplots(figsize=(12, 4))

            ax.plot(windows_2048[0])
            ax.set_xlabel("Sample")
            ax.set_ylabel("Amplitude")
            ax.set_title("Vibration signal")

            st.pyplot(fig)

        st.write("2048 windows : ", windows_2048.shape)
        st.session_state["windows_2048"] = windows_2048
        st.success("windows created succesfully!!")

    if "windows_2048" in st.session_state:
        windows = st.session_state["windows_2048"]

        feature_list = []

        for window in windows:
            time_features = extract_features(window)
            freq_features = extract_frequency_features(window)

            features = {**time_features, **freq_features}
            feature_list.append(features)

            st.session_state["features"] = feature_list

    if "features" in st.session_state:
        feature_df = pd.DataFrame(st.session_state["features"])
        st.dataframe(feature_df.head(), use_container_width=True)

    if st.button("Analyze Signal"):
        st.info("Analysis pipeline will run here.")
