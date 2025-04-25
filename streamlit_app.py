import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import joblib
from model_loader import load_model_and_scaler

model, scaler = load_model_and_scaler()

@st.cache_data
def get_clean_data():
    data = pd.read_csv("data/data_day1.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    feature_labels = list(data.drop(columns='diagnosis').columns)
    input_dict = {}
    for key in feature_labels:
        input_dict[key] = st.sidebar.slider(
            key, float(0), float(data[key].max()), float(data[key].mean()))
    return input_dict

def get_scaled_input(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    return torch.from_numpy(input_scaled).float()

def get_radar_chart(input_dict):
    categories = list(input_dict.keys())[:10]
    values = [input_dict[cat] for cat in categories]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Input Features'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    return fig

def make_prediction(input_tensor):
    model.eval()
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor))
        prob = output.item()
        pred = 1 if prob >= 0.5 else 0
    return pred, prob

def main():
    st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    input_dict = add_sidebar()
    st.title("Breast Cancer Predictor")
    st.write("This app predicts whether a breast mass is benign or malignant based on cytological features.")

    col1, col2 = st.columns([3, 1])
    with col1:
        chart = get_radar_chart(input_dict)
        st.plotly_chart(chart)

    with col2:
        input_tensor = get_scaled_input(input_dict)
        pred, prob = make_prediction(input_tensor)
        st.subheader("Prediction")
        if pred == 0:
            st.write("🟢 **Benign**")
        else:
            st.write("🔴 **Malignant**")
        st.write(f"Confidence: `{prob:.2%}`")

if __name__ == '__main__':
    main()
