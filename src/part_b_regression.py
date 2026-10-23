
from shipment_prep import FEATURES
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def preprocess(input):
    x = input[FEATURES]
    y = input[["delivery_hours"]]

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.3, random_state = 42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5,random_state=42)

    sc = StandardScaler()
    x_train_s =sc.fit_transform(x_train)
    x_test_s = sc.transform(x_test)

    df = pd.DataFrame({
        'metric' : ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean","test_scaled_max_abs_mean"],
        'values' : [len(x_train), len(x_val), len(x_test), np.abs(x_train_s.mean(axis=0)).max() , np.abs(x_test_s.mean(axis=0)).max()]
    })
    df.to_csv("outputs/part_b_checks.csv", index=False)

    depths = [1, 2, 3, 4, 6, 8, 12]
    rows = []

    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(x_train,y_train)
        x_train_pred = dt.predict(x_train)
        x_val_pred = dt.predict(x_val)
        train_rmse = root_mean_squared_error(y_train,x_train_pred)
        val_rmse = root_mean_squared_error(y_val,x_val_pred)
        
        rows.append({'max_depth': d,'train_rmse': train_rmse, 'val_rmse': val_rmse})

    dt_2 = DecisionTreeRegressor(max_depth=None,random_state=42)
    dt_2.fit(x_train,y_train)
    x_train_pred_2 = dt_2.predict(x_train)
    x_val_pred_2 = dt_2.predict(x_val)
    train_rmse_2 = root_mean_squared_error(y_train,x_train_pred_2)
    val_rmse_2 = root_mean_squared_error(y_val,x_val_pred_2)
    rows.append({'max_depth': 'unlimited','train_rmse': train_rmse_2, 'val_rmse': val_rmse_2})
    
    pd.DataFrame(rows).to_csv("outputs/part_b_tree_depth.csv", index=False)

    #MODEL 1
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipeline.fit(x_train,y_train)
    model_1_y_pred = pipeline.predict(x_val)
    model1_mae = mean_absolute_error(y_val,model_1_y_pred)
    model1_mse = mean_squared_error(y_val,model_1_y_pred)
    model1_rmse = root_mean_squared_error(y_val,model_1_y_pred)
    model1_r2 = r2_score(y_val,model_1_y_pred)

    # MODEL 2
    dt2 = DecisionTreeRegressor(max_depth=6, random_state=42)
    dt2.fit(x_train,y_train)
    model_2_y_pred = dt2.predict(x_val)
    model2_mae = mean_absolute_error(y_val, model_2_y_pred)
    model2_mse = mean_squared_error(y_val, model_2_y_pred)
    model2_rmse = root_mean_squared_error(y_val, model_2_y_pred)
    model2_r2 = r2_score(y_val, model_2_y_pred)

    # MODEL 3
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    model_3_y_pred = rf.predict(x_val)
    model3_mae = mean_absolute_error(y_val, model_3_y_pred)
    model3_mse = mean_squared_error(y_val, model_3_y_pred)
    model3_rmse = root_mean_squared_error(y_val, model_3_y_pred)
    model3_r2 = r2_score(y_val, model_3_y_pred)

    # MODEL 4
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,random_state=42)
    gb.fit(x_train,y_train)
    model_4_y_pred = gb.predict(x_val)
    model4_mae = mean_absolute_error(y_val,model_4_y_pred)
    model4_mse = mean_squared_error(y_val,model_4_y_pred)
    model4_rmse = root_mean_squared_error(y_val,model_4_y_pred)
    model4_r2 = r2_score(y_val,model_4_y_pred)

    # MEAN BASELINE
    y_mean = y_train.mean()
    baseline_mean = np.full(len(y_val), y_mean)
    baseline_mae = mean_absolute_error(y_val,baseline_mean)
    baseline_mse = mean_squared_error(y_val,baseline_mean)
    baseline_rmse = root_mean_squared_error(y_val,baseline_mean)
    baseline_r2 = r2_score(y_val,baseline_mean)
    

    rows_to_write = [
        {'model':'mean_baseline', 'split':'val', 'MAE':baseline_mae, 'MSE':baseline_mse, 'RMSE':baseline_rmse, 'R2':baseline_r2},
        {'model':'linear_regression', 'split':'val', 'MAE':model1_mae, 'MSE':model1_mse, 'RMSE':model1_rmse, 'R2':model1_r2},
        {'model':'decision_tree', 'split':'val', 'MAE':model2_mae, 'MSE':model2_mse, 'RMSE':model2_rmse, 'R2':model2_r2},
        {'model':'random_forest', 'split':'val', 'MAE':model3_mae, 'MSE':model3_mse, 'RMSE':model3_rmse, 'R2':model3_r2},
        {'model':'gradient_boosting', 'split':'val', 'MAE':model4_mae, 'MSE':model4_mse, 'RMSE':model4_rmse, 'R2':model4_r2}
    ]

    pd.DataFrame(rows_to_write).to_csv("outputs/part_b_metrics.csv", index=False)

   
    



if __name__ == "__main__":
    modelling = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "modelling_table.csv")
    df = preprocess(modelling)
    