import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.shipment_prep import clean_shipments, build_features


def predict_delay(new_shipments,hubs,model,feature_order,delay_thres,imput_vals,hub_segments):
    median_dict = dict(zip(imput_vals["column"], imput_vals["median"]))
    thres = pd.read_csv(delay_thres)["threshold"].iloc[0]
    cleaned = clean_shipments(new_shipments,median_dict)
    features_df = build_features(cleaned,hubs)
    order = feature_order["feature"].tolist()
    x = features_df[order]
    delay_probability = np.round(model.predict_proba(x)[:,1],4)
    flagged = np.where(delay_probability >= thres , 1,0)
    joined_df = features_df.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")
    joined_df["delay_probability"] = delay_probability
    joined_df["flagged"] = flagged
    
    conditions = [
        (joined_df["flagged"] == 1) & (joined_df["segment"] == "Escalate"),
        (joined_df["flagged"] == 1)
    ]

    comments = ["Escalate to hub manager" , "Proactive customer call"]

    joined_df["action"] = np.select(conditions, comments, default="No action")

    joined_df[["shipment_id", "hub_id" ,"segment", "delay_probability", "flagged", "action"]].to_csv("outputs/new_shipment_actions.csv", index=False)


if __name__ == "__main__":
    new_shipments = Path(__file__).resolve().parent / "data" / "new_shipments.csv"
    hubs = Path(__file__).resolve().parent / "data" / "hubs.csv"
    model = joblib.load("models/delay_model.pkl")
    feature_order = pd.read_csv(Path(__file__).resolve().parent / "models" / "feature_order.csv")
    delay_threshold = Path(__file__).resolve().parent / "models" / "delay_threshold.csv"
    impute_vals = pd.read_csv(Path(__file__).resolve().parent / "models" / "impute_values.csv")
    hub_segments = pd.read_csv(Path(__file__).resolve().parent / "outputs" / "hub_segments.csv")

    predict_delay(new_shipments,hubs,model,feature_order,delay_threshold,impute_vals,hub_segments)