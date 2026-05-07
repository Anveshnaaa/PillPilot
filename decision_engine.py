import pandas as pd


def _is_critically_low(status: object) -> bool:
    cleaned = str(status).strip().lower()
    return "critical" in cleaned or cleaned in {"low", "very low", "critically low"}


def _is_surplus(status: object) -> bool:
    cleaned = str(status).strip().lower()
    return "surplus" in cleaned or cleaned in {"high", "excess"}


def _as_float(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def _resolve_dataframe(csv_path=None, df=None):
    if df is not None:
        return df.copy()
    if csv_path is None:
        raise ValueError("Provide either csv_path or df.")
    return pd.read_csv(csv_path)


def find_best_transfer_or_reorder(
    csv_path=None,
    target_store=None,
    medicine_name=None,
    unit_cost=10.0,
    transport_cost_per_unit=2.0,
    desired_cover_days=7,
    df=None,
):
    if target_store is None or medicine_name is None:
        raise ValueError("target_store and medicine_name are required.")

    df = _resolve_dataframe(csv_path=csv_path, df=df)

    required_cols = [
        "store_name",
        "medicine_name",
        "current_stock",
        "daily_demand",
        "predicted_status",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    med_df = df[df["medicine_name"] == medicine_name].copy()

    if med_df.empty:
        return {
            "store_name": target_store,
            "medicine_name": medicine_name,
            "final_decision": "No Action",
            "reason": f"No data found for {medicine_name}.",
        }

    target_rows = med_df[med_df["store_name"] == target_store]

    if target_rows.empty:
        return {
            "store_name": target_store,
            "medicine_name": medicine_name,
            "final_decision": "No Action",
            "reason": f"{target_store} does not have data for {medicine_name}.",
        }

    target = target_rows.iloc[-1]

    if not _is_critically_low(target["predicted_status"]):
        return {
            "store_name": target_store,
            "medicine_name": medicine_name,
            "final_decision": "No Action",
            "reason": f"{target_store} is not critically low for {medicine_name}.",
            "current_status": target["predicted_status"],
        }

    daily_demand = target["daily_demand"]
    current_stock = target["current_stock"]

    required_stock = daily_demand * desired_cover_days
    needed_units = max(0, int(required_stock - current_stock))

    if needed_units == 0:
        return {
            "store_name": target_store,
            "medicine_name": medicine_name,
            "final_decision": "No Action",
            "reason": "Stock is already enough based on desired cover days.",
        }

    surplus_stores = med_df[
        (med_df["store_name"] != target_store) &
        (med_df["predicted_status"].map(_is_surplus))
    ].copy()

    best_transfer = None
    target_distance = _as_float(target.get("distance_from_distributor_miles"), default=0.0)
    target_unit_price = _as_float(target.get("unit_price"), default=unit_cost)
    target_reorder_fee = _as_float(target.get("reorder_fee"), default=0.0)

    for _, store in surplus_stores.iterrows():
        available_extra = max(
            0,
            int(store["current_stock"] - (store["daily_demand"] * desired_cover_days))
        )

        if available_extra <= 0:
            continue

        transfer_units = min(needed_units, available_extra)

        donor_unit_price = _as_float(store.get("unit_price"), default=target_unit_price)
        donor_distance = _as_float(store.get("distance_from_distributor_miles"), default=target_distance)
        distance_gap = abs(donor_distance - target_distance)
        per_unit_move_cost = max(transport_cost_per_unit, donor_unit_price * 0.05)
        distance_fee = distance_gap * 0.12
        handling_fee = 4.0
        transfer_cost = (transfer_units * per_unit_move_cost) + distance_fee + handling_fee

        if best_transfer is None or transfer_cost < best_transfer["transfer_cost"]:
            best_transfer = {
                "from_store": store["store_name"],
                "to_store": target_store,
                "medicine_name": medicine_name,
                "transfer_units": transfer_units,
                "transfer_cost": round(float(transfer_cost), 2),
                "available_extra": available_extra,
            }

    shipping_fee = target_distance * 0.1
    reorder_cost = (needed_units * target_unit_price) + target_reorder_fee + shipping_fee

    if best_transfer and best_transfer["transfer_cost"] < reorder_cost:
        return {
            "store_name": target_store,
            "final_decision": "Transfer Stock",
            "reason": "Transfer is cheaper than placing a new reorder.",
            "needed_units": needed_units,
            "from_store": best_transfer["from_store"],
            "to_store": best_transfer["to_store"],
            "medicine_name": medicine_name,
            "transfer_units": best_transfer["transfer_units"],
            "transfer_cost": best_transfer["transfer_cost"],
            "reorder_cost": round(float(reorder_cost), 2),
            "savings": round(float(reorder_cost - best_transfer["transfer_cost"]), 2),
        }

    return {
        "final_decision": "Reorder Stock",
        "reason": "No cheaper surplus transfer option was found.",
        "needed_units": needed_units,
        "medicine_name": medicine_name,
        "store_name": target_store,
        "reorder_cost": round(float(reorder_cost), 2),
    }