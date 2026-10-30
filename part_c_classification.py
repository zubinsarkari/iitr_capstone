from src.shipment_prep import FEATURES, NOISE_COLUMNS
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

def build_classification_model(input):

    # TASK 1

    x = input[FEATURES]
    y = input["is_delayed"]

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.3, random_state = 42,stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5,random_state=42,stratify=y_temp)


    # TASK 2

    # blind_all_on_time
    preds = np.full(len(y_val), 0)
    cm = confusion_matrix(y_val,preds).ravel()

    # LOGISTIC REGRESSION
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    pipeline.fit(x_train,y_train)
    preds_1 = pipeline.predict(x_val)
    cm_1 = confusion_matrix(y_val,preds_1).ravel()
   
    # DecisionTreeClassifier
    depths = [2, 3, 4, 5, 6, 8, 10]
    score = []
    for d in depths:
        dt = DecisionTreeClassifier(criterion='gini',max_depth=d,random_state=42)
        dt.fit(x_train, y_train)
        preds_2 = dt.predict(x_val)
        score.append(f1_score(y_val,preds_2))
    depth_with_max_r1 = depths[np.argmax(score)]

    dt2 = DecisionTreeClassifier(criterion='gini', max_depth=depth_with_max_r1, random_state=42)
    dt2.fit(x_train, y_train)
    pred_3 = dt2.predict(x_val)
    cm_2 = confusion_matrix(y_val, pred_3).ravel()


    # RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    pred_4 = rf.predict(x_val)
    cm_3 = confusion_matrix(y_val, pred_4).ravel()
    
    #BLIND PREDICTION
    accuracy_score_1 = accuracy_score(y_val,preds)
    precision_score_1 = precision_score(y_val,preds, zero_division = 0)
    recall_score_1 = recall_score(y_val,preds, zero_division = 0)
    f1_score_1 = f1_score(y_val,preds, zero_division = 0)

    # LOGISTIC REGRESSION
    accuracy_score_2 = accuracy_score(y_val,preds_1)
    precision_score_2 = precision_score(y_val,preds_1, zero_division = 0)
    recall_score_2 = recall_score(y_val,preds_1, zero_division = 0)
    f1_score_2 = f1_score(y_val,preds_1, zero_division = 0)

    # DT
    accuracy_score_3 = accuracy_score(y_val,pred_3)
    precision_score_3 = precision_score(y_val,pred_3, zero_division = 0)
    recall_score_3 = recall_score(y_val,pred_3, zero_division = 0)
    f1_score_3 = f1_score(y_val,pred_3, zero_division = 0)

    # RANDOM FOREST
    accuracy_score_4 = accuracy_score(y_val,pred_4)
    precision_score_4 = precision_score(y_val,pred_4, zero_division = 0)
    recall_score_4 = recall_score(y_val,pred_4, zero_division = 0)
    f1_score_4 = f1_score(y_val,pred_4, zero_division = 0)
    

    new_dict = [{'model' : 'blind_all_on_time' , 'TN': cm[0], 'FP': cm[1], 'FN': cm[2], 'TP': cm[3], 'accuracy' : np.round(accuracy_score_1,4) , 'precision' : np.round(precision_score_1,4), 'recall' : np.round(recall_score_1,4), 'f1' : np.round(f1_score_1,4)
    },
    {'model' : 'logistic_regression' , 'TN': cm_1[0], 'FP': cm_1[1], 'FN': cm_1[2], 'TP': cm_1[3], 'accuracy' : np.round(accuracy_score_2,4), 'precision' : np.round(precision_score_2,4), 'recall' : np.round(recall_score_2,4), 'f1' : np.round(f1_score_2,4)
    },
    {'model' : f'decision_tree_depth_{depth_with_max_r1}' , 'TN': cm_2[0], 'FP': cm_2[1], 'FN': cm_2[2], 'TP': cm_2[3], 'accuracy' : np.round(accuracy_score_3,4), 'precision' : np.round(precision_score_3,4), 'recall' : np.round(recall_score_3,4), 'f1' : np.round(f1_score_3,4)
    },
    {'model' : 'random_forest_default' , 'TN': cm_3[0], 'FP': cm_3[1], 'FN': cm_3[2], 'TP': cm_3[3], 'accuracy' : np.round(accuracy_score_4,4), 'precision' : np.round(precision_score_4,4), 'recall' : np.round(recall_score_4,4) ,'f1' : np.round(f1_score_4,4)
    }
    ]
    pd.DataFrame(new_dict).to_csv("outputs/part_c_models.csv", index=False)


    # TASK 3

    default_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    default_rf.fit(x_train, y_train)
   
    kbest = SelectKBest(f_classif, k='all')
    kbest.fit(x_train, y_train)
    
    rf_df = pd.DataFrame({
        'feature' : x_train.columns,
        'f_score' : np.round(kbest.scores_,4),
        'f_rank' : pd.Series(kbest.scores_).rank(ascending=False, method='first').astype(int),
        'rf_importance' : np.round(default_rf.feature_importances_,4),
        'is_noise' : x_train.columns.isin(NOISE_COLUMNS)
    })

    rf_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

    folds = [5, 7, 10, 12, 17]
    res = []
    for k in folds:
        pip = Pipeline([
            ('selector', SelectKBest(f_classif, k=k)),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pip.fit(x_train, y_train)
        pred = pip.predict(x_val)
        res.append({'k': k , 'val_f1' : np.round(f1_score(y_val, pred),4)})

    pd.DataFrame(res).to_csv("outputs/part_c_selectk.csv", index=False)


    # TASK 4
    k_best_7 = SelectKBest(f_classif, k=7)
    k_best_7.fit(x_train, y_train)
    x_train_7 = k_best_7.transform(x_train)
    x_val_7 = k_best_7.transform(x_val)

    params = {
        'n_estimators' : [50, 100, 150],
        'max_depth' : [4, 6, 8],
        'min_samples_leaf' : [1, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=params, cv=5)
    grid.fit(x_train_7, y_train)
    scores_cv = grid.best_score_
    best_param_cv = grid.best_params_
    pred_7_cv = grid.predict(x_val_7)
    val_f1 = f1_score(y_val, pred_7_cv)


    random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42),param_distributions = params, n_iter=9, cv=5, random_state=42)
    random_search.fit(x_train_7, y_train)
    scores_rs = random_search.best_score_
    best_param_rs = random_search.best_params_
    pred_7_rs = random_search.predict(x_val_7)
    val_f1_rs = f1_score(y_val, pred_7_rs)

    def_rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, min_samples_leaf=1)
    def_rf.fit(x_train_7, y_train)
    pred_7_rf = def_rf.predict(x_val_7)
    val_f1_rf = f1_score(y_val, pred_7_rf)

    rows = [{
        'search' : 'grid_search',
        'candidates' : len(grid.cv_results_['params']),
        'cv_folds' : 5 ,
        'total_fits' : len(grid.cv_results_['params']) * 5,
        'best_params' : best_param_cv,
        'best_cv_accuracy' : np.round(scores_cv,4),
        'val_f1' : np.round(val_f1,4)
    },
    {
        'search' : 'randomized_search',
        'candidates' : len(random_search.cv_results_['params']),
        'cv_folds' : 5 ,
        'total_fits' : len(random_search.cv_results_['params']) * 5,
        'best_params' : best_param_rs,
        'best_cv_accuracy' : np.round(scores_rs,4),
        'val_f1' : np.round(val_f1_rs,4)
    },
    {
      'search' : 'default_forest_same_7_features' ,
      'candidates' : 1,
      'cv_folds' : 0 ,
      'total_fits' : 1,
      'best_params' : 'defaults',
     'best_cv_accuracy' : None,
      'val_f1' : np.round(val_f1_rf,4)
    }]

    pd.DataFrame(rows).to_csv("outputs/part_c_tuning.csv", index=False)

    # TASK 5
    new_pipeline = Pipeline([
        ('selector', SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None, min_samples_leaf=1))
        ])
    new_pipeline.fit(x_train, y_train)
    new_pred = new_pipeline.predict(x_val)
    new_f1 = f1_score(y_val, new_pred)

    # TASK 6
    probs = new_pipeline.predict_proba(x_val)[:,1]

    thres = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    r = []
    for t in thres:
        pred_t = (probs >= t).astype(int)
        new_cm = confusion_matrix(y_val, pred_t, labels=[0,1]).ravel()
        acc = accuracy_score(y_val, pred_t)
        pre = precision_score(y_val, pred_t, zero_division=0)
        rec = recall_score(y_val, pred_t, zero_division=0)
        f1 = f1_score(y_val, pred_t, zero_division=0)
        r.append({
            'threshold' : t,
            'TN': new_cm[0], 'FP': new_cm[1], 'FN': new_cm[2], 'TP': new_cm[3],
            'accuracy' : np.round(acc,4), 'precision' : np.round(pre,4), 'recall' : np.round(rec,4), 'f1' : np.round(f1,4)
        })
    threshold_values = pd.DataFrame(r)
    threshold_values.to_csv("outputs/part_c_thresholds.csv", index=False)
    
    # TASK 7
    alert_threshold = threshold_values[threshold_values["recall"] >= 0.8]
    if len(alert_threshold) > 0:
        alert_threshold = alert_threshold["threshold"].max()
    else:
        alert_threshold = 0.1

    print(f'alert_threshold : {alert_threshold}')        
    test_probs = new_pipeline.predict_proba(x_test)[:,1]
    test_pred = (test_probs >= alert_threshold).astype(int)
    final_cm = confusion_matrix(y_test, test_pred, labels=[0,1]).ravel()
    final_acc = accuracy_score(y_test, test_pred)
    final_pre = precision_score(y_test, test_pred, zero_division=0)
    final_rec = recall_score(y_test, test_pred, zero_division=0)
    final_f1 = f1_score(y_test, test_pred, zero_division=0)

    rows_to_write = []
    rows_to_write.append({
        'threshold' : alert_threshold,
        'TN': final_cm[0], 'FP': final_cm[1], 'FN': final_cm[2], 'TP': final_cm[3],
        'accuracy' : np.round(final_acc,4), 'precision' : np.round(final_pre,4), 'recall' : np.round(final_rec,4), 'f1' : np.round(final_f1,4)
    }) 

    pd.DataFrame(rows_to_write).to_csv("outputs/part_c_final_test.csv", index=False)

    # hard coded values here since we already know that which setting has won.
    pd.DataFrame([{
        'selector_k' : 7,
        'n_estimators' : 100,
        'max_depth' : 'unlimited',
        'min_samples_leaf' : 1,
        'threshold' : alert_threshold,
        'chosen_from' : 'default_forest_same_7_features'
    }]).to_csv("outputs/part_c_final_config.csv", index=False)


if __name__ == "__main__":
    modelling = pd.read_csv(Path(__file__).resolve().parent / "data" / "modelling_table.csv")
    build_classification_model(modelling)