# Pill Pilot

Pill Pilot is an AI-driven pharmacy inventory intelligence platform built with:

- Vanilla JavaScript frontend (`static/`)
- Flask backend (`web_backend.py`)
- Machine learning models for inventory classification and demand forecasting
- Live FDA recall intelligence using openFDA APIs
- NIH RxNorm integration for medicine interaction checks
- RAG-style assistant workflows for grounded operational guidance

The current demo flow uses:

- `web_backend.py`
- `static/index.html`

---

# Features

- Inventory classification:
  - Low Stock
  - Optimal Stock
  - Surplus Stock

- 3-day medicine demand forecasting

- Reorder vs transfer operational recommendations

- Live FDA recall and shortage intelligence

- Drug interaction checking using NIH RxNorm APIs

- AI assistant with evidence-backed responses

---

# Tech Stack

## Frontend
- HTML
- CSS
- Vanilla JavaScript

## Backend
- Flask

## ML / AI
- scikit-learn
- RandomForestClassifier
- RandomForestRegressor
- TF-IDF
- cosine similarity
- Levenshtein fuzzy matching

## APIs
- openFDA API
- NIH RxNorm API

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone https://github.com/Anveshnaaa/PillPilot
cd PillPilot
```

---

## 2. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 4. Train ML Models

```bash
python train_and_save_models.py
```

This creates trained model files inside the `models/` folder.

---

## 5. Run Backend Server

```bash
python web_backend.py
```

---

## 6. Open Website

Open browser and go to:

```text
http://127.0.0.1:8000
```

---

# How To Use

1. Open the app
2. Select role:
   - Distributor
   - Store Owner
3. Upload inventory CSV
4. Click `Run Analysis`
5. Explore:
   - Inventory tables
   - 3-day forecasting
   - Recommendation queue
   - FDA recall findings
   - Drug interaction checks
   - AI assistant workflows

---

# Main Workflows

## ML Workflow

1. Upload inventory CSV
2. RandomForestClassifier predicts:
   - Low Stock
   - Optimal
   - Surplus
3. RandomForestRegressor forecasts future demand
4. Decision engine recommends:
   - Reorder
   - Transfer

---

## Assistant Workflow

1. User asks question
2. Intent detection classifies request
3. Runtime inventory facts are gathered
4. Relevant information is retrieved using TF-IDF + cosine similarity
5. Assistant generates evidence-backed response

---

## FDA Recall Workflow

1. Backend retrieves live FDA recalls from openFDA
2. Inventory medicines are matched using:
   - NDC exact match
   - TF-IDF similarity
   - cosine similarity
   - Levenshtein fuzzy matching
   - lot number matching
3. System returns:
   - Findings
   - Risk level
   - Recommended action
   - Supporting evidence

---

# Active API Endpoints

- `GET /api/health`
- `POST /api/analyze`
- `POST /api/medicine-insights`
- `POST /api/live-fda-insights`
- `POST /api/rxnorm-interaction`
- `POST /api/ask-anything-hybrid`
- `POST /api/distributor-chat`

---

# Required CSV Columns

## Inference CSV

- `store_name`
- `medicine_name`
- `date`
- `current_stock`
- `daily_demand`
- `expiry_days_left`
- `season`
- `category`
- `demand_trend`
- `lead_time_days`
- `unit_price`
- `reorder_fee`
- `distance_from_distributor_miles`

---

## Training CSV

Must also include:

- `inventory_status`

---

# Demo CSV

Use one of the included sample CSV files for testing/demo.

Example:

```text
inventory_sample_4.csv
```

---

# Internet Requirement

Internet connection is required for:

- Live FDA recall checks
- NIH RxNorm interaction checks

---

# Notes

- FDA recall intelligence uses live openFDA retrieval
- Interaction checks use NIH RxNorm APIs
- Forecasting uses recursive lag-based prediction
- RAG workflows use TF-IDF + cosine similarity retrieval
