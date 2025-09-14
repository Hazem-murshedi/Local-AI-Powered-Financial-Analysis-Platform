# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 12:01:42 2025

@author: hazem
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.title("AI-Powered Trading Platform")

# Sidebar for inputs
st.sidebar.header("Prediction Settings")
last_vol = st.sidebar.number_input("Last Volatility", value=0.015)

# Load IBM stock data
df = pd.read_csv("ibm_preprocessed.csv")
data = pd.read_csv("IBM.csv")
df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)

SEQUENCE_LENGTH =10


# Show last 10 rows (used for prediction)
last_rows = df.iloc[-240:-230]
st.subheader(" Last 10 Rows (used for prediction)")
st.dataframe(last_rows)

# Candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=data["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Adj_Close"]
)])
fig.update_layout(title="IBM Stock Candlestick")
st.plotly_chart(fig)

# Prepare last 10 rows as features
features = last_rows.to_dict(orient="records")
payload = {"features": features, "last_volatility": last_vol}

# Call FastAPI when button is pressed
if st.button("Run Prediction"):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f" Price Prediction: {result['price_prediction']:.18f}")
            st.write("Volatility Forecast:", result["volatility_forecast"])
            st.write("Value at Risk (VaR):", result["value_at_risk"])
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Connection failed: {e}")
