import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib 
import plotly.express as px
from collections import Counter
import importlib

from preprocessing.MatLoader import load_mat_file
from preprocessing.windowing import create_windows
from preprocessing.time_features import extract_features
from preprocessing.frequency_features import extract_frequency_features
from preprocessing.image_processing import process_window, create_cnn_input

from models.CNN_2D import load_model_cnn2d as load_2d_model, predict_cnn2d as predict_2d
from models.msdc_dense import MSDCDenseTCN 
from models.CNN_1D import FaultDetectionCNN
from models.LSTM import BearingLSTM
from models.TCN_ATTENTION import TCNAttention

@st.cache_resource
def load_ml_models():
    xgb_model = joblib.load("models/xgb_combined_model.joblib") 
    rf_model = joblib.load("models/rf_combined_model.joblib") 
    svm_model = joblib.load("models/svm_combined_model.joblib") 
    svm_scaler = joblib.load("models/svm_scaler.joblib") 
    return xgb_model, rf_model, svm_model, svm_scaler

@st.cache_resource
def load_dl_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_1d = FaultDetectionCNN(num_classes=4).to(device) 
    checkpoint_1d = torch.load("models/mixed_fault_detection_1dcnn.pth", map_location=device)
    state_dict_1d = checkpoint_1d.get("model_state_dict", checkpoint_1d)
    model_1d.load_state_dict(state_dict_1d)
    model_1d.eval()

    model_tcn = MSDCDenseTCN(num_classes=4).to(device) 
    model_tcn.load_state_dict(torch.load("models/best_mixed_msdc_densetcn (1).pth", map_location=device)) 
    model_tcn.eval()

    model_lstm = BearingLSTM(num_classes=4).to(device)
    model_lstm.load_state_dict(torch.load("models/Raw_CWRU_MFPT_best_LSTM_weights.pth", map_location=device))
    model_lstm.eval()

    model_ta = TCNAttention(num_classes=4).to(device)
    model_ta.load_state_dict(torch.load("models/Raw_CWRU_MFPT_best_TCN&Attention_weights.pth", map_location=device))
    model_ta.eval()
    
    return model_1d, model_tcn, model_lstm, model_ta, device

@st.cache_resource
def load_2d_cnn():
    return load_2d_model("models/Raw_2channel_best_model.pth") 

def run_dl_inference(model, device, windows_array):
    inputs = torch.tensor(windows_array, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()

def run_dl_inference2(model, device, windows_array):
    windows_array = windows_array[:, :, np.newaxis]
    inputs = torch.tensor(windows_array, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()

cnn2d_label_map = {
    0: "Ball",
    1: "Inner",
    2: "Normal",
    3: "Outer"
}

label_map = {
    0: "Normal",
    1: "Ball Fault",
    2: "Inner race",
    3: "Outer race"
}

tcn_label_map = {
    0: "Normal",
    1: "Inner race",
    2: "Outer race",
    3: "Ball Fault"
}

def standardize_2d_label(val):
    val_str = str(val).strip().lower()
    if "normal" in val_str or "healthy" in val_str:
        return "Normal"
    elif "ball" in val_str:
        return "Ball Fault"
    elif "inner" in val_str:
        return "Inner race"
    elif "outer" in val_str:
        return "Outer race"
    return "Unknown"

st.set_page_config(page_title="Bearing Fault Diagnosis", layout="wide")
st.title("Bearing Fault Diagnosis Platform")
st.write("Upload a raw vibration signal `.mat` file to run multi-model inference.")

uploaded_file = st.file_uploader("Upload MATLAB (.mat) file", type=["mat"])

if uploaded_file is not None:
    signal = load_mat_file(uploaded_file)
    st.success(f"Loaded: `{uploaded_file.name}` (Signal shape: {signal.shape})")

    windows_2048 = create_windows(signal, window_size=2048) 
    windows_1024 = create_windows(signal, window_size=1024, step=1024) 

    st.subheader("1. Signal Visualization")
    if windows_2048 is not None and len(windows_2048) > 0:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(windows_2048[0], color="#1f77b4", linewidth=0.8)
        ax.set_xlabel("Sample Point")
        ax.set_ylabel("Amplitude")
        ax.set_title("Vibration Signal Sample (Window Size: 2048)")
        st.pyplot(fig)

    if len(windows_1024) > 0:
        first_gasf, first_gadf = process_window(windows_1024[0]) 
        
        with st.expander("Preview 2D GAF Transformation (Window Size: 1024)", expanded=False):
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(first_gasf, caption="GASF (Gramian Angular Summation Field)", use_container_width=True)
            with col_img2:
                st.image(first_gadf, caption="GADF (Gramian Angular Difference Field)", use_container_width=True)

    st.subheader("2. Extracted Features (for ML Models)")
    with st.spinner("Extracting time & frequency features..."):
        feature_list = []
        for window in windows_2048:
            t_feat = extract_features(window)
            f_feat = extract_frequency_features(window)
            feature_list.append({**t_feat, **f_feat})
        feature_df = pd.DataFrame(feature_list)

    st.dataframe(feature_df.head(), use_container_width=True)
    st.caption(f"Total 2048-Windows: {len(windows_2048)} | Total 1024-Windows: {len(windows_1024)}")

    st.divider()

    if st.button("🚀 Analyze Signal", type="primary"):
        progress_bar = st.progress(0, text="Initializing model pipelines...")

        progress_bar.progress(10, text="Loading models...")
        xgb_model, rf_model, svm_model, svm_scaler = load_ml_models() 
        cnn_1d_model, tcn_model, lstm_model, ta_model, dl_device = load_dl_models() 
        cnn_2d_model, dl_2d_device = load_2d_cnn()

        progress_bar.progress(30, text="Running XGBoost, Random Forest, and SVM...")
        scaled_features = svm_scaler.transform(feature_df) 
        xgb_preds = xgb_model.predict(feature_df)
        rf_preds = rf_model.predict(feature_df)
        svm_preds = svm_model.predict(scaled_features)

        progress_bar.progress(60, text="Running 1D CNN and DenseTCN...")
        cnn1d_preds = run_dl_inference(cnn_1d_model, dl_device, windows_2048)
        tcn_preds = run_dl_inference(tcn_model, dl_device, windows_2048)
        lstm_preds = run_dl_inference2(lstm_model, dl_device, windows_2048)
        ta_preds = run_dl_inference2(ta_model, dl_device, windows_2048)

        progress_bar.progress(80, text="Running 2D CNN on GAF representations...")
        cnn2d_raw_preds = []
        for w in windows_1024:
            gasf_img, gadf_img = process_window(w)
            cnn_in = create_cnn_input(gasf_img, gadf_img)
            pred, _ = predict_2d(cnn_2d_model, dl_2d_device, cnn_in)
            cnn2d_raw_preds.append(pred)

        progress_bar.progress(100, text="Aggregating predictions...")
        progress_bar.empty()

        model_results = {
            "XGBoost": [label_map.get(int(p), "Unknown") for p in xgb_preds],
            "Random Forest": [label_map.get(int(p), "Unknown") for p in rf_preds],
            "SVM": [label_map.get(int(p), "Unknown") for p in svm_preds],
            "1D CNN": [label_map.get(int(p), "Unknown") for p in cnn1d_preds],
            "DenseTCN": [tcn_label_map.get(int(p), "Unknown") for p in tcn_preds], 
            "2D CNN": [standardize_2d_label(p) for p in cnn2d_raw_preds],
            "LSTM" : [label_map.get(int(p), "Unknown") for p in lstm_preds],
            "TCN AND ATTENTION" : [label_map.get(int(p), "Unknown") for p in ta_preds]
        }

        st.subheader("Model Diagnostic Breakdown")
        cols = st.columns(len(model_results))

        all_preds = []
        for col, (model_name, preds) in zip(cols, model_results.items()):
            all_preds.extend(preds)
            pred_counts = Counter(preds)
            majority_fault = pred_counts.most_common(1)[0][0]
            confidence = (pred_counts.most_common(1)[0][1] / len(preds)) * 100

            with col:
                st.markdown(f"**{model_name}**")
                if "Normal" in majority_fault:
                    st.success(majority_fault)
                else:
                    st.error(majority_fault)
                st.caption(f"Agreement: {confidence:.1f}%")

        global_counts = Counter(all_preds)
        final_diagnosis = global_counts.most_common(1)[0][0]
        final_confidence = (global_counts.most_common(1)[0][1] / len(all_preds)) * 100

        st.divider()
        st.subheader("Final System Diagnosis")

        if "Normal" in final_diagnosis:
            st.success(f"### Final Diagnosis: **{final_diagnosis}** (Confidence: {final_confidence:.1f}%)")
        else:
            st.error(f"### Final Diagnosis: **{final_diagnosis}** (Confidence: {final_confidence:.1f}%)")

        count_df = pd.DataFrame(list(global_counts.items()), columns=["Fault Type", "Aggregated Votes"])
        fig = px.pie(
            count_df,
            values="Aggregated Votes",
            names="Fault Type",
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)