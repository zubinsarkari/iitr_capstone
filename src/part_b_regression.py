
from shipment_prep import FEATURES
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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
        'value' : [len(x_train), len(x_val), len(x_test), np.abs(x_train_s.mean(axis=0)).max() , np.abs(x_test_s.mean(axis=0)).max()]
    })
    df.to_csv("outputs/part_b_checks.csv", index=False)

    depths = [1, 2, 3, 4, 6, 8, 12]
    rows = []

    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(x_train,y_train)
        x_train_pred = dt.predict(x_train)
        x_val_pred = dt.predict(x_val)
        train_rmse = round(root_mean_squared_error(y_train,x_train_pred),4)
        val_rmse = round(root_mean_squared_error(y_val,x_val_pred),4)
        
        rows.append({'max_depth': d,'train_rmse': train_rmse, 'val_rmse': val_rmse})

    dt_2 = DecisionTreeRegressor(max_depth=None,random_state=42)
    dt_2.fit(x_train,y_train)
    x_train_pred_2 = dt_2.predict(x_train)
    x_val_pred_2 = dt_2.predict(x_val)
    train_rmse_2 = round(root_mean_squared_error(y_train,x_train_pred_2),4)
    val_rmse_2 = round(root_mean_squared_error(y_val,x_val_pred_2),4)
    rows.append({'max_depth': 'unlimited','train_rmse': train_rmse_2 , 'val_rmse': val_rmse_2})
    
    pd.DataFrame(rows).to_csv("outputs/part_b_tree_depth.csv", index=False)

    #MODEL 1 - LINEAR REGRESSION
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipeline.fit(x_train,y_train)
    model_1_y_pred = pipeline.predict(x_val)
    model1_mae =  round(mean_absolute_error(y_val,model_1_y_pred),4)
    model1_mse =  round(mean_squared_error(y_val,model_1_y_pred),4)
    model1_rmse = round(root_mean_squared_error(y_val,model_1_y_pred),4)
    model1_r2 =   round(r2_score(y_val,model_1_y_pred),4)

    # MODEL 2 - DT
    dt2 = DecisionTreeRegressor(max_depth=6, random_state=42)
    dt2.fit(x_train,y_train)
    model_2_y_pred = dt2.predict(x_val)
    model2_mae = round(mean_absolute_error(y_val, model_2_y_pred),4)
    model2_mse = round(mean_squared_error(y_val, model_2_y_pred),4)
    model2_rmse = round(root_mean_squared_error(y_val, model_2_y_pred),4)
    model2_r2 = round(r2_score(y_val, model_2_y_pred),4)

    # MODEL 3 - RANDOM FOREST
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    model_3_y_pred = rf.predict(x_val)
    model3_mae = round(mean_absolute_error(y_val, model_3_y_pred),4)
    model3_mse = round(mean_squared_error(y_val, model_3_y_pred),4)
    model3_rmse = round(root_mean_squared_error(y_val, model_3_y_pred),4)
    model3_r2 = round(r2_score(y_val, model_3_y_pred),4)

    # MODEL 4 - GB
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,random_state=42)
    gb.fit(x_train,y_train)
    model_4_y_pred = gb.predict(x_val)
    model4_mae = round(mean_absolute_error(y_val,model_4_y_pred),4)
    model4_mse = round(mean_squared_error(y_val,model_4_y_pred),4)
    model4_rmse = round(root_mean_squared_error(y_val,model_4_y_pred),4)
    model4_r2 = round(r2_score(y_val,model_4_y_pred),4)

    # MEAN BASELINE
    y_mean = y_train.mean()
    baseline_mean = np.full(len(y_val), y_mean)
    baseline_mae = round(mean_absolute_error(y_val,baseline_mean),4)
    baseline_mse = round(mean_squared_error(y_val,baseline_mean),4)
    baseline_rmse = round(root_mean_squared_error(y_val,baseline_mean),4)
    baseline_r2 = round(r2_score(y_val,baseline_mean),4)


    rows_to_write = [
                {'model':'mean_baseline', 'split':'val', 'MAE':baseline_mae, 'MSE':baseline_mse, 'RMSE':baseline_rmse, 'R2':baseline_r2},
                {'model':'linear_regression', 'split':'val', 'MAE':model1_mae, 'MSE':model1_mse, 'RMSE':model1_rmse, 'R2':model1_r2},
                {'model':'decision_tree', 'split':'val', 'MAE':model2_mae, 'MSE':model2_mse, 'RMSE':model2_rmse, 'R2':model2_r2},
                {'model':'random_forest', 'split':'val', 'MAE':model3_mae, 'MSE':model3_mse, 'RMSE':model3_rmse, 'R2':model3_r2},
                {'model':'gradient_boosting', 'split':'val', 'MAE':model4_mae, 'MSE':model4_mse, 'RMSE':model4_rmse, 'R2':model4_r2}]

    #Learning-rate experiment
    gb_rows = []
    for lr in [0.01, 0.1, 1.0]:
            gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr,random_state=42)
            gb.fit(x_train,y_train)
            y_pred = gb.predict(x_val)
            gb_rmse = root_mean_squared_error(y_val,y_pred)
            gb_r2 = r2_score(y_val,y_pred)
            gb_rows.append({'learning_rate' : lr, 'val_rmse' : np.round(gb_rmse,4), 'val_r2': np.round(gb_r2,4)})

    pd.DataFrame(gb_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)   


    # CROSS VALIDATION
    df = pd.DataFrame(rows_to_write)
    condition = df[(df["split"] == "val") & (df["model"] != "mean_baseline")]
    df_sorted= condition.sort_values("RMSE", ascending=True).head(2)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    x = pd.concat([x_train,x_val])
    y = pd.concat([y_train,y_val])

    models = {
    "linear_regression": pipeline,
    "decision_tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)    
    }

    top2_models = df_sorted["model"].tolist()

    cv_rows = []    
    for ml_model in top2_models:
        scores = cross_val_score(models[ml_model], x, y, cv=kf)
        cv_rows.append({'model':ml_model, 'mean_r2': np.round(scores.mean(),4) , 'std_r2': np.round(scores.std(),4)})

    pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv",index=False)

    lr_y_test_pred = pipeline.predict(x_test)
    lr_test_mae =  round(mean_absolute_error(y_test,lr_y_test_pred),4)
    lr_test_mse =  round(mean_squared_error(y_test,lr_y_test_pred),4)
    lr_rmse = round(root_mean_squared_error(y_test,lr_y_test_pred),4)
    lr_r2 =   round(r2_score(y_test,lr_y_test_pred),4)

    metric_rows = rows_to_write + [
            {'model':'linear_regression', 'split':'test', 'MAE':lr_test_mae, 'MSE':lr_test_mse, 'RMSE':lr_rmse, 'R2':lr_r2}
        ]
    
    pd.DataFrame(metric_rows).to_csv("outputs/part_b_metrics.csv", index=False)
       

if __name__ == "__main__":
    modelling = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "modelling_table.csv")
    preprocess(modelling)
    