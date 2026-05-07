# Pill Pilot

Pill Pilot is a pharmacy inventory and safety dashboard built with:

- a vanilla JS frontend (`static/`)
- a Flask backend (`web_backend.py`)
- ML models for stock status and demand forecasting
- live FDA and RxNorm integrations for recall/interaction checks

The current demo flow is `web_backend.py` + `static/index.html`.

## Quick Start

Run these commands from the project root:

```bash
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python train_and_save_models.py
./venv/bin/python web_backend.py
```

Then open:

`http://127.0.0.1:8000`

## How To Use The Demo

1. Open the app and pick a role (Distributor or Store Owner).
2. Upload a CSV with the required schema.
3. Click `Run Analysis`.
4. Explore:
   - inventory tables
   - 3-day forecast
   - action/recommendation queue
   - safety and recall findings
   - Pharmacy Assistant (Ask Anything + Recall/Interaction/Inventory tabs)

## Active API Endpoints

- `GET /api/health`
- `POST /api/analyze`
- `POST /api/medicine-insights`
- `POST /api/live-fda-insights`
- `POST /api/rxnorm-interaction`
- `POST /api/ask-anything-hybrid`
- `POST /api/distributor-chat`

## Required CSV Columns (Inference)

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

## Training Note

For model training (`train_and_save_models.py`), the CSV must also include:

- `inventory_status`
