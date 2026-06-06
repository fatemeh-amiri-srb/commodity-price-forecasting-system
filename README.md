# Commodity Price Forecasting System

A production-style machine learning system for forecasting global oil prices using return-based modeling, macroeconomic indicators, and market structure features.

---

## 📌 Project Overview

Commodity markets are highly volatile and influenced by a combination of:

- Global energy prices
- Agricultural commodity markets
- Currency fluctuations
- Macroeconomic conditions

This project builds a **leakage-free, time-series forecasting pipeline** to model short-term price dynamics using machine learning.

Instead of predicting raw prices, the system models **log-returns**, making the problem statistically stable and more realistic.

---

## 🧠 Key Idea

> Predicting commodity prices directly is unstable due to non-stationarity.  
> This system predicts **log returns** and reconstructs prices from them.

---

## ⚙️ Methodology

### 1. Data Processing
- Time-series alignment
- Missing value handling
- Numeric normalization

### 2. Feature Engineering (No Leakage)
- Log returns of oil, palm, and crude markets
- Lagged return features (1–3 months)
- Volatility estimation (rolling std)
- Momentum indicators
- Price spread features (relative valuation)
- Macro proxy signals (FX changes)

### 3. Target Variable
y(t+1) = log(P(t+1)) - log(P(t))
---

### 4. Regime Features
Market regimes are approximated using volatility thresholds:

- Low volatility regime
- High volatility regime
- Crisis regime

---

### 5. Model
- XGBoost Regressor
- Time-series cross validation
- Walk-forward evaluation

---

## 📊 Model Performance

| Metric | Value |
|------|------|
| MAE | 0.0587 |
| RMSE | 0.0711 |
| Baseline (Naive) | 0.0644 |

> The model slightly outperforms the naive baseline, indicating meaningful signal extraction from noisy commodity data.

---

## 🔥 Key Insights

- Commodity markets exhibit **strong short-term autocorrelation**
- Return-based modeling significantly improves stability
- Macro features alone are not sufficient predictive drivers
- Regime features have limited standalone predictive power but improve robustness

---

## 📈 Feature Importance (Top Drivers)

- Lagged oil returns (1–3 months)
- Current oil return
- Momentum signals
- Palm and crude oil spread dynamics
- Volatility features

---

## 🚀 Forecast Example

The system generates recursive multi-step forecasts:
[1613.33 → 1648.54 → 1622.30]
These represent reconstructed price trajectories from predicted returns.

---

## 🏗️ Project Structure
```
commodity-price-forecasting-system/
│
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│
├── main.py
├── requirements.txt
├── README.md
```
---

## 🧪 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Time-Series Cross Validation

---

## 📌 How to Run

```bash
pip install -r requirements.txt
python main.py
```
👤 Author

Fatemeh Amiri
AI Engineer | Computer Science Student 

⸻

⭐ If you like this project

Give it a star ⭐ and feel free to explore improvements or contribute.
