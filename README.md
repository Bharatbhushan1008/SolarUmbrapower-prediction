# SolarUmbrapower-prediction
AolarUmbrapower-prediction is an AI-powered IoT project designed to predict and optimize the energy performance of solar-powered smart umbrellas. Using Deep Learning (LSTM networks), it forecasts the next-step battery level (%) based on environmental and operational data such as irradiance, temperature, and power usage.
🌞 Umbra – Smart Solar Umbrella Ecosystem with AI-Based Energy Prediction
⚙️ Deep Learning | IoT | Streamlit | Sustainability
🧭 Overview

Umbra is a Deep Learning–powered IoT energy prediction system designed to optimize energy usage and battery management for solar-powered smart umbrellas.

It combines real solar generation and weather data with simulated IoT umbrella behavior to predict future battery levels using an LSTM (Long Short-Term Memory) model.

A Streamlit dashboard provides real-time visualization of power generation, consumption, and predictive analytics for each umbrella unit.

🚀 Features

✅ Data Cleaning & Integration:

Merges real solar plant data and weather sensor readings using time-based alignment.

Handles missing, noisy, and mismatched timestamps automatically.

✅ IoT Simulation:

Generates realistic umbrella behavior (power usage, humidity, wind speed, etc.).

Produces multi-feature time-series data for AI model training.

✅ Deep Learning (LSTM) Model:

Predicts next-step battery percentage based on environmental and operational factors.

Learns temporal energy trends (charge/discharge cycles).

✅ Streamlit Web Dashboard:

Interactive UI for selecting umbrella unit & sequence length.

Displays recent telemetry and next-step battery prediction.

Visualizes past battery trends and predicted future value.

✅ Sustainable Tech:

Promotes efficient energy management in solar-powered IoT systems.

🧱 Project Structure
Umbra/
│
├── data/
│   ├── Plant_1_Generation_Data.csv
│   └── Plant_1_Weather_Sensor_Data.csv
│
├── output/
│   ├── Cleaned_Plant1_Data.csv
│   ├── Umbra_IoT_Simulated.csv
│   ├── X_train.npy, X_test.npy, y_train.npy, y_test.npy
│
├── models/
│   ├── umbra_lstm_best.h5
│   ├── scaler_features.pkl
│   └── scaler_target.pkl
│
├── src/
│   ├── data_clean.py
│   ├── simulate_umbra.py
│   ├── train.py
│
├── app/
│   └── streamlit_app.py
│
├── Umbra_pipeline.ipynb      # Full Jupyter notebook workflow
├── requirements.txt          # Dependencies list
└── README.md

🧠 Technologies Used
Category	Tools / Frameworks
Programming Language	Python 3.11
Data Handling	Pandas, NumPy
Machine Learning	Scikit-learn
Deep Learning	TensorFlow / Keras
Visualization	Matplotlib
Deployment	Streamlit
Environment	VS Code
🔍 Model Architecture
Input → LSTM(128, return_sequences=True)
      → Dropout(0.2)
      → LSTM(64)
      → Dense(32, activation='relu')
      → Dense(1, activation='linear')
Output → Predicted Battery Level (%)


Optimizer: Adam

Loss: Mean Squared Error (MSE)

Metrics: Mean Absolute Error (MAE)

🧩 How It Works

Data Cleaning (data_clean.py)

Reads solar generation & weather CSVs.

Parses and aligns timestamps using merge_asof.

Outputs Cleaned_Plant1_Data.csv.

IoT Simulation (simulate_umbra.py)

Simulates 25 umbrella units with power, temperature, and humidity readings.

Creates Umbra_IoT_Simulated.csv.

Model Training (train.py)

Splits, scales, and trains an LSTM model on time-series data.

Saves model & scalers to /models.

Streamlit App (streamlit_app.py)

Loads the trained model and scalers.

Lets user select umbrella unit & sequence window.

Predicts next-step battery percentage and visualizes trends.

🌐 Streamlit App Demo
Run locally:
streamlit run app/streamlit_app.py

Dashboard Features:

Select umbrella unit (#1–#25)

Adjust sequence length (e.g., 24 → last 6 hours)

See latest telemetry data

View predicted next-step battery level

Interactive trend chart with real-time updates

📈 Example Output

Predicted Battery Level (Next Step): 70.15%

🧮 Evaluation Metrics
Metric	Description	Typical Value
MAE	Mean Absolute Error	~2–5%
RMSE	Root Mean Square Error	<5%
R² Score	Prediction Accuracy	0.95–0.99
🌍 Real-World Applications

Smart solar umbrella installations in campuses & public spaces.

Predictive battery analytics for IoT devices.

Renewable energy optimization in microgrids.

Smart city energy management systems.

📜 Research Insight

Umbra demonstrates how AI can enhance sustainability by predicting solar-powered device behavior.
The LSTM model captures real-world temporal dependencies between solar irradiance, power generation, temperature, and battery usage, enabling predictive control and improved energy efficiency in IoT ecosystems.

📦 Installation

Clone Repository

git clone https://github.com/yourusername/Umbra-Solar-IoT.git
cd Umbra-Solar-IoT


Create Virtual Environment

python -m venv .venv
.venv\Scripts\activate       # on Windows


Install Dependencies

pip install -r requirements.txt


Run Streamlit App

streamlit run app/streamlit_app.py

⚙️ Requirements

requirements.txt includes:

numpy>=1.23
pandas>=1.5
scikit-learn>=1.3
matplotlib>=3.7
tensorflow>=2.10
streamlit>=1.37
protobuf>=4.25
h5py>=3.8

A Streamlit web dashboard provides an intuitive interface where users can:

Select any umbrella unit

Adjust the historical sequence length (timesteps)

Visualize recent telemetry data and predicted battery levels

Monitor energy usage and charging trends interactively

The system is designed to support sustainable and data-driven energy management. By leveraging predictive analytics, Umbra can help anticipate low battery conditions, optimize charging schedules, and ensure consistent performance across solar-powered IoT devices.

Key Features:

Real-world solar and weather data cleaning and merging

IoT data simulation for multiple umbrella units

LSTM-based battery prediction model

Interactive visualization with Streamlit

Real-time forecasting of solar energy performance

Tech Stack:
Python, TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit

Applications:

Smart city solar infrastructures

Renewable energy optimization

IoT-based power management systems

Predictive maintenance for solar devices

Goal:
Umbra demonstrates how AI and IoT can work together to create smarter, more efficient, and sustainable solar ecosystems — turning ordinary solar umbrellas into intelligent, self-managing energy devices for the future.
