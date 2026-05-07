import pandas as pd


def find_best_transfer_or_reorder(
    csv_path,
    target_store,
    medicine_name,
    unit_cost=10,
    transport_cost_per_unit=2,
    desired_cover_days=7
):
    df = pd.read_csv(csv_path)

    required_cols = [
        "store_name",
        "medicine_name",
        "current_stock",
        "daily_demand",
        "predicted_status"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    med_df = df[df["medicine_name"] == medicine_name].copy()

    if med_df.empty:
        return {
            "final_decision": "No Action",
            "reason": f"No data found for {medicine_name}."
        }

    target_rows = med_df[med_df["store_name"] == target_store]

    if target_rows.empty:
        return {
            "final_decision": "No Action",
            "reason": f"{target_store} does not have data for {medicine_name}."
        }

    target = target_rows.iloc[-1]

    if target["predicted_status"] != "Critically Low":
        return {
            "final_decision": "No Action",
            "reason": f"{target_store} is not critically low for {medicine_name}.",
            "current_status": target["predicted_status"]
        }

    daily_demand = target["daily_demand"]
    current_stock = target["current_stock"]

    required_stock = daily_demand * desired_cover_days
    needed_units = max(0, int(required_stock - current_stock))

    if needed_units == 0:
        return {
            "final_decision": "No Action",
            "reason": "Stock is already enough based on desired cover days."
        }

    surplus_stores = med_df[
        (med_df["store_name"] != target_store) &
        (med_df["predicted_status"] == "Surplus")
    ].copy()

    best_transfer = None

    for _, store in surplus_stores.iterrows():
        available_extra = max(
            0,
            int(store["current_stock"] - (store["daily_demand"] * desired_cover_days))
        )

        if available_extra <= 0:
            continue

        transfer_units = min(needed_units, available_extra)
        transfer_cost = transfer_units * transport_cost_per_unit

        if best_transfer is None or transfer_cost < best_transfer["transfer_cost"]:
            best_transfer = {
                "from_store": store["store_name"],
                "to_store": target_store,
                "medicine_name": medicine_name,
                "transfer_units": transfer_units,
                "transfer_cost": transfer_cost,
                "available_extra": available_extra
            }

    reorder_cost = needed_units * unit_cost

    if best_transfer and best_transfer["transfer_cost"] < reorder_cost:
        return {
            "final_decision": "Transfer Stock",
            "reason": "Transfer is cheaper than placing a new reorder.",
            "needed_units": needed_units,
            "from_store": best_transfer["from_store"],
            "to_store": best_transfer["to_store"],
            "medicine_name": medicine_name,
            "transfer_units": best_transfer["transfer_units"],
            "transfer_cost": best_transfer["transfer_cost"],
            "reorder_cost": reorder_cost,
            "savings": reorder_cost - best_transfer["transfer_cost"]
        }

    return {
        "final_decision": "Reorder Stock",
        "reason": "No cheaper surplus transfer option was found.",
        "needed_units": needed_units,
        "medicine_name": medicine_name,
        "store_name": target_store,
        "reorder_cost": reorder_cost
    }