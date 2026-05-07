from pathlib import Path

import joblib
import pandas as pd

from ml_pipeline import train_demand_model, train_status_model, validate_input_schema


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_CSV = PROJECT_ROOT / "inventory_large.csv"
MODEL_DIR = PROJECT_ROOT / "models"


def main() -> None:
    if not TRAINING_CSV.exists():
        raise FileNotFoundError(f"Training CSV not found: {TRAINING_CSV}")

    # Train once, then reuse models at runtime. The website never retrains.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAINING_CSV)

    # require_target=True ensures training data includes inventory_status label.
    validate_input_schema(df, require_target=True)

    # ML Model 1: inventory status classifier (predicts critical/optimal/surplus-type status).
    status_model, status_metadata = train_status_model(df)

    # ML Model 2: demand forecaster built from lag features.
    demand_model, demand_metadata = train_demand_model(df)

    # Persist artifacts so Flask API can load and infer without retraining.
    joblib.dump(status_model, MODEL_DIR / "status_model.pkl")
    joblib.dump(status_metadata, MODEL_DIR / "status_metadata.pkl")
    joblib.dump(demand_model, MODEL_DIR / "demand_model.pkl")
    joblib.dump(demand_metadata, MODEL_DIR / "demand_metadata.pkl")

    print("Saved model artifacts:")
    print(f" - {MODEL_DIR / 'status_model.pkl'}")
    print(f" - {MODEL_DIR / 'status_metadata.pkl'}")
    print(f" - {MODEL_DIR / 'demand_model.pkl'}")
    print(f" - {MODEL_DIR / 'demand_metadata.pkl'}")
    print(f"Status model accuracy: {status_metadata['metrics']['accuracy']:.4f}")
    print(f"Demand model MAE: {demand_metadata['metrics']['mae']:.4f}")


if __name__ == "__main__":
    main()
