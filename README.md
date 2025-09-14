# 📈 AI-Powered Financial Analysis & Trading Platform

This project implements a local AI-powered financial analysis and trading platform. It uses deep learning (LSTM) for market prediction, GARCH for risk assessment metrics like volatility forecasting, and computes Value at Risk (VaR) — all **without cloud-based APIs**. A FastAPI backend serves predictions, and a Streamlit dashboard provides interactive data visualization.

---

## 🚀 Features

- **Data Preprocessing**
  - Uses IBM stock data (CSV)
  - Features: High, Low, Adj_Close, Volume, SMA, High_Low_Diff
  - Scaling with MinMaxScaler

- **Local AI Inference**
  - **LSTM Model:** Pre-trained PyTorch model for forecasting `log_return` and predicting the next day's stock price
  - **GARCH Model:** Statistical model for volatility prediction and VaR computation

- **Decoupled Architecture**
  - **FastAPI Backend:** RESTful API to serve AI model predictions locally
  - **Streamlit Dashboard:** Interactive UI for input, data visualization, and displaying results

---

## 💻 Technical Stack

- **Backend:** Python, FastAPI, PyTorch, ARCH
- **Frontend:** Python, Streamlit, Plotly, Pandas
- **Testing:** Postman

---

## ⚙️ Installation & Usage

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.x
- pip

### 2. Clone the Repository

```bash
git clone https://github.com/Hazem-murshedi/Local-AI-Powered-Financial-Analysis-Platform.git
cd Local-AI-Powered-Financial-Analysis-Platform
```

### 3. Install Python Dependencies

```bash
pip install fastapi uvicorn arch pydantic pandas torch streamlit requests plotly
```

### 4. Place the Model Files

Ensure the following pre-trained model files are in the project’s root directory:

- `lstm_stock_prediction.pt`
- `garch_model_results.pkl`

You can generate these files by running the provided [Google Colab notebook](https://colab.research.google.com/drive/1Jr1Gi6zVBcEvk5LNGBRrbgqiO91DAejw).

### 5. Run the Backend Server

Start the FastAPI server:

```bash
python -m uvicorn app:app --reload
```

### 6. Run the Streamlit Dashboard

Open a new terminal and run the Streamlit app:

```bash
streamlit run dashboard.py
```

This will open the dashboard in your browser at [http://localhost:8501](http://localhost:8501).

---

## 🧠 Technical Architecture & Design Choices

- **Local-First Philosophy:** All core functionalities are performed on-device.
- **Decoupled Architecture:** FastAPI backend serves as a dedicated API layer, separating AI logic from the frontend. API endpoints were tested with Postman before integrating with Streamlit.
- **On-Device AI Inference:** The FastAPI server loads pre-trained LSTM and GARCH models into memory on startup, enabling low-latency predictions and risk assessment without internet/cloud APIs.
- **Hybrid AI Approach:** Combines a deep learning LSTM model for `log_return` forecasting with a GARCH statistical model for risk assessment, offering a robust analysis.
- **Streamlit for User Experience:** Rapidly builds an interactive, visually rich dashboard, allowing focus on core AI functionality with a professional, data-centric UI.

---

## 👨‍💻 Author

Hazem Ahmed Murshedi
