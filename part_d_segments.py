import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from src.shipment_prep import FEATURES, compute_medians, clean_shipments

def build_final_model(input):

    # TASK 1
    df = (input.groupby(input["hub_id"]).agg(
            shipments = ("shipment_id","count"),
            mean_delivery_hours = ("delivery_hours", "mean"),
            delay_rate = ('is_delayed',"mean"),
            mean_distance_km = ("distance_km", 'mean'),
            mean_n_stops = ("n_stops", 'mean')
        ).reset_index().sort_values("hub_id")
        )
    df.to_csv("outputs/hub_profile.csv", index=False)
        
    # TASK 2
    x = df[["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]]
    sc = StandardScaler()
    x_scaled = sc.fit_transform(x)
    kmeans= KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(x_scaled)
    df["cluster"] = kmeans.labels_

    # TASK 3
    clusters = [2, 3, 4, 5]
    out = []
    for k in clusters:
        kmeans_1 = KMeans(n_clusters= k, random_state=42, n_init=10)
        kmeans_1.fit(x_scaled)
        labels = kmeans_1.labels_

        within = []
        across = []

        for i in range(24):
            for j in range(i + 1, 24):
                d = np.sqrt(np.sum((x_scaled[i] - x_scaled[j]) ** 2))
                if labels[i] == labels[j]:
                    within.append(d)
                else:
                    across.append(d)    


        sizes = sorted(Counter(labels).values(), reverse=True)
        across_over_within = np.mean(across) / np.mean(within)
        out.append({
            'K' : k,
            'across_over_within' : across_over_within,
            'cluster_sizes' : "-".join(str(s) for s in sizes)
        })

    pd.DataFrame(out).to_csv("outputs/part_d_separation.csv", index=False)
                  
    # TASK 4
    cluster_means = df.groupby("cluster")['delay_rate'].mean()
    order = cluster_means.sort_values().index.tolist()
    names = ["Standard", "Watch", "Escalate"]
    mapping = dict(zip(order,names))
    df["segment"] = df["cluster"].map(mapping)

    hubs = pd.read_csv(Path(__file__).resolve().parent / "data" / "hubs.csv")
    df2 = df.merge(hubs, on='hub_id', how='left')
    df2[["hub_id","segment", "hub_city", "hub_tier"]].to_csv("outputs/hub_segments.csv", index=False)
    
    df3 = (
        df.groupby('segment').agg(
            hubs = ('hub_id', 'count'),
            mean_delivery_hours = ('mean_delivery_hours', 'mean'),
            delay_rate = ('delay_rate', 'mean'),
            mean_distance_km = ('mean_distance_km' , 'mean'),
            mean_n_stops = ('mean_n_stops', 'mean')
        )
     ).reset_index()

    df3.to_csv("outputs/segment_profile.csv", index=False)

    dict_map  = {1: "Standard", 2: "Watch", 3: "Escalate"}
    df2["temp_segment"] = df2["hub_tier"].map(dict_map)
    mismatched_df = df2[df2["segment"] != df2["temp_segment"]]
    mismatched_df = mismatched_df.drop(columns="temp_segment")
    mismatched_df[["hub_id","segment", "hub_city", "hub_tier"]].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

    # TASK 5
    final_config = pd.read_csv("outputs/part_c_final_config.csv")
    n_esm = int(final_config["n_estimators"].iloc[0])
    min_samples = int(final_config["min_samples_leaf"].iloc[0])
    features = int(final_config["selector_k"].iloc[0])
    depth_val = final_config["max_depth"].iloc[0]
    depth = None if depth_val == "unlimited" else int(depth_val)

    x = input[FEATURES]
    y = input["is_delayed"]
    
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.3, random_state = 42,stratify=y)
    
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5,random_state=42,stratify=y_temp)
    

    model = Pipeline([
        ('selector', SelectKBest(f_classif, k=features)),
        ("model", RandomForestClassifier(random_state=42, n_estimators=n_esm, max_depth=depth, min_samples_leaf=min_samples))
        ])

    model.fit(x_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,"models/delay_model.pkl")

    pd.DataFrame({'threshold': [final_config["threshold"].iloc[0]]}).to_csv("models/delay_threshold.csv", index=False)

    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

    shipments = Path(__file__).resolve().parent / "data" / "raw_shipments.csv"
    cleaned = clean_shipments(shipments)
    medians = compute_medians(cleaned)

    pd.DataFrame({
        'column' : list(medians.keys()),
        'median' : list(medians.values())
    }).to_csv("models/impute_values.csv", index=False)

    reloaded_model = joblib.load("models/delay_model.pkl")
    probs = model.predict_proba(x_test)[:,1]
    reloaded_probs = reloaded_model.predict_proba(x_test)[:,1]
    reload_max_abs_probability_diff = abs(probs - reloaded_probs).max()


    pd.DataFrame({
        "metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
        "value" : [len(x_test), reload_max_abs_probability_diff, final_config["threshold"].iloc[0]]
    }).to_csv("outputs/part_d_reload_check.csv", index=False)





if __name__ == "__main__":
    modelling = pd.read_csv(Path(__file__).resolve().parent / "data" / "modelling_table.csv")
    build_final_model(modelling)