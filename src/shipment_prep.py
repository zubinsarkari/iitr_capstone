
import pandas as pd
import numpy as np
from pathlib import Path  

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw, medians=None):
    raw_data = Path(__file__).resolve().parent.parent / "data" / raw
    raw_df = pd.read_csv(raw_data)
    deduplicated_df = raw_df.drop_duplicates()
    deduplicated_df["service_type"] = deduplicated_df["service_type"].str.strip().str.title()
    deduplicated_df["payment_mode"] = deduplicated_df["payment_mode"].fillna("Unknown")
    deduplicated_df["declared_value"] = deduplicated_df["declared_value"].str.replace('₹','').str.replace(",","")
    deduplicated_df["declared_value_inr"] = pd.to_numeric(deduplicated_df["declared_value"], errors='coerce').fillna(0).astype(int)
    deduplicated_df["dispatch_time"] = pd.to_datetime(deduplicated_df["dispatch_time"])
    median_distance = round(deduplicated_df["distance_km"].median(),4)
    median_weight = round(deduplicated_df["parcel_weight_kg"].median(),4)
    median_value_inr= round(deduplicated_df["declared_value_inr"].median(),4)


    deduplicated_df["parcel_weight_kg"] = deduplicated_df["parcel_weight_kg"].where(deduplicated_df["parcel_weight_kg"] <= 100)

    if medians is not None:
        deduplicated_df["distance_km"] = deduplicated_df["distance_km"].fillna(medians.get('distance_km',0))
        deduplicated_df["parcel_weight_kg"] = deduplicated_df["parcel_weight_kg"].fillna(medians.get('parcel_weight_kg',0))
        deduplicated_df["declared_value_inr"] = deduplicated_df["declared_value_inr"].fillna(medians.get('declared_value_inr',0))
    else:    
        deduplicated_df["distance_km"] = deduplicated_df["distance_km"].fillna(median_distance)
        deduplicated_df["parcel_weight_kg"] = deduplicated_df["parcel_weight_kg"].fillna(median_weight)
        deduplicated_df["declared_value_inr"] = deduplicated_df["declared_value_inr"].fillna(median_value_inr)
    
    cleaned_df = deduplicated_df.drop(columns='declared_value')
    return cleaned_df


def build_features(clean,hubs):
    hubs_data = Path(__file__).resolve().parent.parent / "data" / hubs
    hubs_df = pd.read_csv(hubs_data)
    joined_df = clean.merge(hubs_df,on = ['hub_id'], how='left')
    joined_df["promised_hours"] = joined_df['service_type'].map(PROMISED_HOURS)
    joined_df["is_cod"] = np.where(joined_df["payment_mode"] == "COD", 1, 0)
    joined_df["is_weekend"] = np.where(joined_df["dispatch_time"].dt.dayofweek >= 5, 1, 0)
    joined_df["dispatch_hour"] = joined_df["dispatch_time"].dt.hour
    joined_df["log_declared_value"] = np.log(joined_df["declared_value_inr"])
    return joined_df

#joined_df[["dispatch_time","dispatch_hour","log_declared_value"]][joined_df["shipment_id"] == 'SHP10034']
a = build_features(clean_shipments("raw_shipments.csv"),"hubs.csv")

print(a.head(5))