import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES, compute_medians, clean_shipments
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

def package_model(data, modelling_data):
        input = pd.read_csv(data)
        n_esm = int(input["n_estimators"].iloc[0])
        min_samples = int(input["min_samples_leaf"].iloc[0])
        features = int(input["selector_k"].iloc[0])
        depth_val = input["max_depth"].iloc[0]
        depth = None if depth_val == "unlimited" else int(depth_val)
    
        x = modelling_data[FEATURES]
        y = modelling_data["is_delayed"]
        
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.3, random_state = 42,stratify=y)
        
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5,random_state=42,stratify=y_temp)
        
    
        model = Pipeline([
            ('selector', SelectKBest(f_classif, k=features)),
            ("model", RandomForestClassifier(random_state=42, n_estimators=n_esm, max_depth=depth, min_samples_leaf=min_samples))
            ])
    
        model.fit(x_train, y_train)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model,"models/delay_model.pkl")
    
        pd.DataFrame({'threshold': [input["threshold"].iloc[0]]}).to_csv("models/delay_threshold.csv", index=False)
    
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
            "value" : [len(x_test), reload_max_abs_probability_diff, input["threshold"].iloc[0]]
        }).to_csv("outputs/part_d_reload_check.csv", index=False)


if __name__ == "__main__":
    modelling_data = pd.read_csv(Path(__file__).resolve().parent / "data" / "modelling_table.csv")
    final_config = Path(__file__).resolve().parent / "outputs" / "part_c_final_config.csv"
    package_model(final_config, modelling_data)