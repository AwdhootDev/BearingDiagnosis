import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.MatLoader import load_mat_file
from preprocessing.windowing import create_windows
from preprocessing.time_features import extract_features
from preprocessing.frequency_features import extract_frequency_features

from collections import Counter
import joblib  
import plotly.express as px

@st.cache_resource
def load_xgb_model():
    MODEL_PATH = "models\\xgb_model.joblib" 
    model = joblib.load(MODEL_PATH)
    return model

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
        st.info("Running XGBoost analysis .......")

        feature_df = pd.DataFrame(st.session_state["features"])
        model = load_xgb_model()

        predictions = model.predict(feature_df)

        label_map = {
                    0: "Normal",
                    1: "Ball Fault",
                    2: "Inner race",
                    3: "Outer race"
                }

        mapped_preds = [label_map.get(int(pred), f"Unknown Fault ({pred})") for pred in predictions]
        pred_counts = Counter(mapped_preds)
        majority_diagnosis = pred_counts.most_common(1)[0][0]
        confidence = (pred_counts.most_common(1)[0][1] / len(predictions)) * 100

        st.subheader("Diagnosis Results")
        if "Normal" in majority_diagnosis:
                st.success(f"Final Diagnosis: **{majority_diagnosis}** (Confidence: {confidence:.1f}%)")
        else:
            st.error(f"Final Diagnosis: **{majority_diagnosis}** (Confidence: {confidence:.1f}%)")

        pred_df = pd.DataFrame(mapped_preds, columns=["Predicted Class"])
        st.write("### Prediction Distribution Across Windows")

        count_df = pd.DataFrame(list(pred_counts.items()), columns=["Fault Type", "Number of Windows"])

        fig = px.pie(
            count_df, 
            values="Number of Windows", 
            names="Fault Type",
            hole=0.5, # This creates the donut hole
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))

        st.plotly_chart(fig, use_container_width=True)