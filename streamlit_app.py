# app/streamlit_app.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silence TF INFO logs

from pathlib import Path
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

PROJECT = Path.cwd()
OUT_DIR = PROJECT / 'output'
MODEL_DIR = PROJECT / 'models'

DATA_FP = OUT_DIR / 'Umbra_IoT_Simulated.csv'
MODEL_FP = MODEL_DIR / 'umbra_lstm_best.h5'   # or umbra_lstm_final.h5
SCALER_FEAT = MODEL_DIR / 'scaler_features.pkl'
SCALER_TGT = MODEL_DIR / 'scaler_target.pkl'

st.set_page_config(page_title="Umbra — Battery Prediction", layout="wide")
st.title("Umbra — DL Energy Prediction (LSTM)")

# --- Basic checks ---
if not DATA_FP.exists():
    st.error("Simulated Umbra data not found. Run the notebook or pipeline to create output/Umbra_IoT_Simulated.csv")
    st.stop()

df = pd.read_csv(DATA_FP, parse_dates=['timestamp'])
units = sorted(df['umbrella_id'].unique().tolist())
unit = st.sidebar.selectbox("Select umbrella unit", units)
seq_len = st.sidebar.slider("Sequence length", 6, 72, 24)

sub = df[df['umbrella_id'] == unit].sort_values('timestamp').reset_index(drop=True)
st.subheader(f"Recent telemetry for umbrella #{unit}")
st.dataframe(sub.tail(8))

feature_cols = ['irradiance','ambient_temp','module_temp','power_generated_W',
                'power_used_W','battery_level_%','hour','day_of_week']

if not MODEL_FP.exists():
    st.warning("Trained model not found. Run the training pipeline to create models/umbra_lstm_best.h5 or umbra_lstm_final.h5")
    st.stop()

# load model with compile=False to avoid deserializing training-only objects (metrics etc.)
try:
    model = tf.keras.models.load_model(str(MODEL_FP), compile=False)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load scalers
try:
    with open(SCALER_FEAT, 'rb') as f:
        feat_scaler = pickle.load(f)
    with open(SCALER_TGT, 'rb') as f:
        tgt_scaler = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load scalers: {e}")
    st.stop()

if len(sub) < seq_len:
    st.warning(f"Not enough history for selected sequence length: have {len(sub)}, need {seq_len}.")
    st.stop()

# prepare window
last_window = sub[feature_cols].tail(seq_len).values.astype(np.float32)

try:
    last_scaled = feat_scaler.transform(last_window)
except Exception as e:
    st.error(f"Feature scaler transform failed: {e}")
    st.stop()

X_input = last_scaled.reshape(1, last_scaled.shape[0], last_scaled.shape[1])

# predict
try:
    pred_scaled = model.predict(X_input, verbose=0).flatten()
    # ensure correct shape for inverse_transform
    pred = tgt_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
    pred = float(pred)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

st.metric("Predicted battery level (next step)", f"{pred:.2f}%")

# Plot history + predicted point
fig, ax = plt.subplots(figsize=(9, 3))
history_vals = sub['battery_level_%'].tail(seq_len).values.astype(np.float32)
ax.plot(range(-len(history_vals), 0), history_vals, marker='o', label='battery history')
ax.scatter(0, pred, marker='X', s=100, color='red', label='predicted next')
ax.set_xlabel('Timesteps (past → 0 predicted)')
ax.set_ylabel('Battery level (%)')
ax.legend()
ax.grid(alpha=0.2)
st.pyplot(fig)
