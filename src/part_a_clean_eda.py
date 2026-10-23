from pathlib import Path
from shipment_prep import build_features , clean_shipments, FEATURES
import numpy as np
import pandas as pd

def modelling_table(raw_data, hubs_data):
    cleaned_df = clean_shipments(raw_data)
    features_df = build_features(cleaned_df,hubs_data)
    features_df["is_delayed"] = np.where(features_df["delivery_hours"] > features_df["promised_hours"],1,0)
    return features_df


def checks(raw_data,modeling_data):
    raw_rows = pd.read_csv(raw_data)
    duplicate_rows_removed = raw_rows.duplicated().sum()
    rows_after_dedup = raw_rows.drop_duplicates()
    missing_distance_km_before = rows_after_dedup["distance_km"].isna().sum()
    missing_parcel_weight_before = rows_after_dedup["parcel_weight_kg"].isna().sum()
    sentinel_weight_rows = (rows_after_dedup["parcel_weight_kg"] > 100.0).sum()
    missing_declared_value_before = rows_after_dedup["declared_value"].isna().sum()
    missing_payment_mode_before = rows_after_dedup["payment_mode"].isna().sum()
    service_type_spellings_before = rows_after_dedup["service_type"].nunique()
    service_type_spellings_after = rows_after_dedup["service_type"].str.strip().str.title().nunique()
    nulls_in_features_and_target = modeling_data[FEATURES + ["delivery_hours","is_delayed"]].isna().sum().sum()
    declared_value_mean_inr = round(np.mean(modeling_data["declared_value_inr"]),2)
    declared_value_median_inr = round(np.median(modeling_data["declared_value_inr"]),1)
    log_declared_value_mean = round(np.mean(modeling_data["log_declared_value"]),3)
    log_declared_value_median = round(np.median(modeling_data["log_declared_value"]),3)
    median_distance_km = modeling_data["distance_km"].median()
    hours_q1 = np.percentile(modeling_data["delivery_hours"],25)
    hours_q3 = np.percentile(modeling_data["delivery_hours"],75)
    iqr = hours_q3 - hours_q1
    hours_lower_bound = hours_q1 - (1.5 * iqr)
    hours_upper_bound = hours_q3 + (1.5 * iqr)
    median_parcel_weight_kg = raw_rows["parcel_weight_kg"].median()
    median_declared_value_inr = modeling_data["declared_value_inr"].median()
    rows_after_hub_merge = len(modeling_data)
    delayed_count = (modeling_data["is_delayed"] == 1).sum()
    hours_outlier_count = np.sum(modeling_data["delivery_hours"] > hours_upper_bound)
    delay_rate = np.round(delayed_count / rows_after_hub_merge, 4)

    
    return pd.DataFrame([{
        'raw_rows' : len(raw_rows),
        'duplicated_rows_removed' : duplicate_rows_removed,
        'rows_after_dedup' : len(rows_after_dedup),
        'missing_distance_km_before' : missing_distance_km_before,
        'missing_parcel_weight_before' : missing_parcel_weight_before,
        'sentinel_weight_rows' : sentinel_weight_rows,
        'missing_declared_value_before' : missing_declared_value_before,
        'missing_payment_mode_before' : missing_payment_mode_before,
        'service_type_spellings_before' : service_type_spellings_before,
        'service_type_spellings_after' : service_type_spellings_after,
        'declared_value_mean_inr' : declared_value_mean_inr,
        'log_declared_value_mean' : log_declared_value_mean,
        'median_distance_km' : median_distance_km,
        'median_parcel_weight_kg' : median_parcel_weight_kg,
        'median_declared_value_inr' : median_declared_value_inr,
        'rows_after_hub_merge' : rows_after_hub_merge,
        'nulls_in_features_and_target' : nulls_in_features_and_target,
        'delayed_count' : delayed_count,
        'delay_rate' : delay_rate,
        'hours_q1 / hours_q3' : f'{round(hours_q1,2)} / {round(hours_q3,2)}',
        'hours_lower_bound / hours_upper_bound' : f'{round(hours_lower_bound,2)} / {round(hours_upper_bound,2)}',
        'hours_outlier_count' : hours_outlier_count,
        'declared_value_median_inr' : declared_value_median_inr,
        'log_declared_value_median' : log_declared_value_median
    }])

def evidence(modeling_table):
    df = modeling_table.copy()
    if "distance_band" not in df.columns:
        df["distance_band"] = pd.cut(
            df["distance_km"],
            bins=[0, 200, 400, 600, 2000],
            labels=["0-200", "201-400", "401-600", "601+"],
        )
    dimensions = ["service_type","hub_tier","distance_band"]
    parts = []
    for dim in dimensions:
        t = (df.groupby(dim).agg(
            shipments = ("is_delayed","size"),
            delayed = ("is_delayed", "sum")
        ).reset_index()
        .rename(columns={dim: "group"}))
        t.insert(0, 'dimension',dim)
        parts.append(t)
    final_df = pd.concat(parts, ignore_index=True)
    final_df["delay_rate"] = np.round(final_df["delayed"] / final_df["shipments"], 4)
    return final_df

def correlations(modeling_table):
    df = modeling_table.corr(numeric_only = True)
    return df

if __name__ == "__main__":
    raw_shipments = Path(__file__).resolve().parent.parent / "data" / "raw_shipments.csv"
    hubs = Path(__file__).resolve().parent.parent / "data" / "hubs.csv"
    new_shipments = Path(__file__).resolve().parent.parent / "data" / "new_shipments.csv"
    modeling_df = modelling_table(raw_shipments,hubs)
    checks_df = checks(raw_shipments,modeling_df)
    evidence_df = evidence(modeling_df)
    correlation_df = correlations(modeling_df)
    top3_df = correlation_df.copy()
    top3_df["abs"] = top3_df[["delivery_hours"]].drop("delivery_hours").abs()
    top3 = np.round(top3_df.sort_values("abs",ascending=False).head(3),4)
    modeling_df.to_csv("data/modelling_table.csv", index=False)
    checks_df.to_csv("data/part_a_checks.csv", index=False)
    evidence_df.to_csv("data/part_a_delay_by_group.csv", index=False)
    correlation_df.to_csv("data/part_a_correlations.csv", index=False)

    mean_before_transform = np.round(modeling_df["declared_value_inr"].mean(),4)
    mean_after_transform = np.round(modeling_df["log_declared_value"].mean(),4)
    median_before_transform = np.round(modeling_df["declared_value_inr"].median(),4)
    median_after_transform = np.round(modeling_df["log_declared_value"].median(),4)


    new_shipments_df = clean_shipments(new_shipments,{'distance_km': 32, 'parcel_weight_kg' : 22,'declared_value_inr' : 330})
    new_build_features = build_features(new_shipments_df,hubs)
    na = new_build_features.isna().sum()
    print(f'null values in new shipment features : {na[na > 0].sum()}') 
    print(f'mean_before_transform : {mean_before_transform}')
    print(f'mean_after_transform : {mean_after_transform}')
    print(f'median_before_transform : {median_before_transform}')
    print(f'median_after_transform : {median_after_transform}')
    
    
