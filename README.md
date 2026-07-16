# 🇵🇰 Pakistan Price Pulse

**Smart Inflation Monitoring & Bazar Advisor Dashboard**

A full-stack data analytics project that tracks, visualizes, and predicts
grocery & fuel prices across 15 major Pakistani cities.

---

## 📌 Project Overview

Pakistan Price Pulse is an interactive analytics dashboard built with
**Python + Streamlit** that helps citizens and policymakers monitor
price inflation, identify the cheapest cities to buy essentials,
and get ML-powered buying recommendations.

### Key Features

| Feature | Description |
|---|---|
| 🏠 Main Dashboard | KPIs, price trends, risk gauge, city ranking |
| 📊 Deep Analytics | Correlation heatmap, volatility ranking, city clusters |
| 🤖 ML Forecasting | Linear regression + Exp. smoothing, anomaly detection |
| 💡 Key Insights | Auto-generated narrative findings from the data |
| 🧾 Budget Calculator | Monthly grocery cost estimator with savings analysis |

---

## ⚠️ Note on Data

This project uses **static/synthetically generated data** for demonstration
purposes (see `data/prices.csv`), since real-time price data from Pakistan
Bureau of Statistics (PBS) is not available via public API.

The synthetic dataset was carefully validated against real published price
ranges (PBS Weekly Sensitive Price Indicator & OGRA fuel notifications) to
ensure realistic values. Data can be regenerated anytime using `generate_data.py`.

The full analytical pipeline (dashboard, ML models, forecasting) is built to
work identically with real-time data whenever a live API becomes available.

---

## 🗂️ Project Structure

```
Pakistanpricepulse/
├── app/
│   └── dashboard/
│       ├── dashboard.py      # Main Streamlit app (5 pages)
│       ├── analytics.py      # ML & statistics engine
│       └── README.md
├── data/
│   └── prices.csv            # Dataset: 15 cities × 15 items × 30 days
├── generate_data.py          # Script to regenerate synthetic data
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## 📊 Dataset

- **15 Cities:** Karachi, Lahore, Islamabad, Faisalabad, Rawalpindi,
  Multan, Peshawar, Quetta, Sialkot, Hyderabad, Gujranwala, Sukkur,
  Bahawalpur, Abbottabad, Mirpur
- **15 Items:** Flour, Sugar, Rice, Milk, Eggs, Petrol, Diesel,
  Onion, Tomato, Potato, Chicken, Meat, Cooking Oil, Tea, Bread
- **5 Categories:** Essentials, Fuel, Vegetables, Protein, Others
- **Time Range:** January 2026 (daily snapshots)
- **Total Records:** ~6,750 rows

---

## 🤖 Machine Learning Models

1. **Linear Regression** — Predicts next price point using OLS on historical data
2. **Exponential Smoothing Forecast** (ARIMA-style) — 5-step ahead forecasts
3. **IsolationForest** — Flags unusual price spikes/drops (anomaly detection)
4. **K-Means Clustering** — Groups cities into Expensive / Moderate / Affordable

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher installed on your computer
- pip (Python package manager, comes with Python)

### Steps to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/SidraaArshaD/Pakistan-Price-Pulse-and-Smart-Bazar-advisor.git
cd Pakistan-Price-Pulse-and-Smart-Bazar-advisor
```

**2. (Optional but recommended) Create a virtual environment**
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r app/dashboard/requirements.txt
```

**4. Run the dashboard**
```bash
python -m streamlit run app/dashboard/dashboard.py
```

**5. Open in browser**
The app will automatically open at:
```
http://localhost:8501
```
If it doesn't open automatically, copy that link into your browser.

---

## 📈 Analytics Features

- **Correlation Heatmap** — Seaborn heatmap showing how item prices co-move across cities
- **Volatility Ranking** — Coefficient of variation per item
- **City Cost Index** — Overall price level comparison across all 15 cities
- **Monthly Inflation Rate** — (Last price − First price) / First price × 100
- **Household Impact** — Extra monthly cost vs national average
- **Budget Calculator** — Enter monthly quantities, see total cost and potential savings

---

## 🛠️ Tech Stack

`Python` · `Streamlit` · `Pandas` · `NumPy` · `Scikit-learn` · `Plotly` · `Seaborn` · `Matplotlib`

---

## 👩‍💻 Developer

**Sidra Arshad** · Big Data Analytics Final Project · 2026

---
