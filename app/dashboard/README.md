# 🇵🇰 Pakistan Price Pulse

**Smart Inflation Monitoring & Bazar Advisor Dashboard**

A full-stack data analytics project that tracks, visualizes, and predicts
grocery & fuel prices across 15 major Pakistani cities.

---

## 📌 Project Overview

Pakistan Price Pulse is an interactive analytics dashboard built with
**Python + Streamlit** that helps citizens and policymakers monitor
real-time price inflation, identify the cheapest cities to buy essentials,
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

## 🗂️ Project Structure

```
Pakistanpricepulse/
├── app/
│   └── dashboard/
│       ├── dashboard.py      # Main Streamlit app (5 pages)
│       └── analytics.py      # ML & statistics engine
├── data/
│   └── prices.csv            # Dataset: 15 cities × 15 items × 30 days
├── generate_data.py          # Script to regenerate synthetic data
├── requirements.txt          # Python dependencies
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

## 📊 Data Sources & Validation

Although this project uses a synthetically generated dataset to demonstrate
the full analytical pipeline, **all price ranges are validated against
real published sources:**

| Item | Our Range | PBS Jan 2026 Range | Source |
|---|---|---|---|
| Flour (1kg) | Rs 154–167 | Rs 150–170 | PBS Weekly SPI |
| Sugar (1kg) | Rs 139–152 | Rs 135–155 | PBS Weekly SPI |
| Rice (1kg) | Rs 146–159 | Rs 140–165 | PBS Weekly SPI |
| Milk (1L) | Rs 240–255 | Rs 235–260 | PBS Weekly SPI |
| Eggs (dozen) | Rs 300–325 | Rs 290–330 | PBS Weekly SPI |
| Petrol (1L) | Rs 272 | Rs 272 (exact) | OGRA Notification |
| Diesel (1L) | Rs 244 | Rs 244 (exact) | OGRA Notification |
| Chicken (1kg) | Rs 508–542 | Rs 500–560 | PBS Weekly SPI |
| Onion (1kg) | Rs 60–76 | Rs 55–80 | PBS Weekly SPI |
| Tomato (1kg) | Rs 68–92 | Rs 60–100 | PBS Weekly SPI |
| Cooking Oil (1L) | Rs 468–488 | Rs 460–500 | PBS Weekly SPI |
| Tea (250g) | Rs 1620–1662 | Rs 1600–1700 | Market Survey |
| Bread (1 loaf) | Rs 62–72 | Rs 60–75 | PBS Weekly SPI |
| Potato (1kg) | Rs 48–62 | Rs 45–70 | PBS Weekly SPI |
| Meat (1kg) | Rs 1078–1128 | Rs 1050–1150 | PBS Weekly SPI |

> **Note:** Petrol and Diesel prices are **exact** government-notified
> rates by OGRA for January 2026. All other items fall within
> PBS Weekly Sensitive Price Indicator reported ranges.

### Why Synthetic Data?
Real-time PBS data is not available via public API.
The synthetic generator replicates realistic price fluctuation patterns
while staying within validated market ranges — making the analytical
pipeline 100% applicable to real data when available.
## 🤖 Machine Learning Models

### 1. Linear Regression (Price Prediction)
Predicts next price point using ordinary least squares on historical data.

### 2. Exponential Smoothing Forecast (ARIMA-style)
Combines exponential smoothing (α = 0.4) with a linear trend component
to produce 5-step ahead forecasts.

### 3. IsolationForest (Anomaly Detection)
Flags unusual price spikes or drops using an ensemble of isolation trees
(contamination = 0.15).

### 4. K-Means Clustering (City Segmentation)
Groups cities into Expensive / Moderate / Affordable clusters based on
average price and price volatility (standard deviation).

---

## ⚙️ Installation & Setup

```bash
# 1. Clone / extract the project
cd Pakistanpricepulse

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
python -m streamlit run app/dashboard/dashboard.py
```

The app will open at **http://localhost:8501**

---

## 📈 Analytics Features

- **Correlation Heatmap** — Seaborn heatmap showing how item prices
  co-move across cities
- **Volatility Ranking** — Coefficient of variation per item
- **City Cost Index** — Overall price level comparison across all 15 cities
- **Monthly Inflation Rate** — (Last price − First price) / First price × 100
- **Household Impact** — Extra monthly cost vs national average
- **Budget Calculator** — Enter monthly quantities, see total cost
  and potential savings by buying from cheaper cities

---

## 👩‍💻 Developer

**Sidra** · Big Data Analytics Final Project · 2026

---