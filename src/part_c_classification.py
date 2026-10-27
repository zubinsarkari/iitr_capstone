from shipment_prep import FEATURES
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def build_classification_model(input):
    x = input[FEATURES]
    y = input["is_delayed"]

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.3, random_state = 42,stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5,random_state=42,stratify=y_temp)


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
    print(f'decision_tree_depth_{depth_with_max_r1}')

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
    

    new_dict = [{'model' : 'blind_all_on_time' , 'TN': cm[0], 'FP': cm[1], 'FN': cm[2], 'TP': cm[3], 'accuracy' :accuracy_score_1, 'precision' : precision_score_1, 'recall' : recall_score_1, 'f1' : f1_score_1
    },
    {'model' : 'logistic_regression' , 'TN': cm_1[0], 'FP': cm_1[1], 'FN': cm_1[2], 'TP': cm_1[3], 'accuracy' :accuracy_score_2, 'precision' : precision_score_2, 'recall' : recall_score_2, 'f1' : f1_score_2
    },
    {'model' : f'decision_tree_depth_{depth_with_max_r1}' , 'TN': cm_2[0], 'FP': cm_2[1], 'FN': cm_2[2], 'TP': cm_2[3], 'accuracy' :accuracy_score_3, 'precision' : precision_score_3, 'recall' : recall_score_3, 'f1' : f1_score_3
    },
    {'model' : 'random_forest_default' , 'TN': cm_3[0], 'FP': cm_3[1], 'FN': cm_3[2], 'TP': cm_3[3], 'accuracy' :accuracy_score_4, 'precision' : precision_score_4, 'recall' : recall_score_4 ,'f1' : f1_score_4
    }
    ]
    pd.DataFrame(new_dict).to_csv("outputs/part_c_models.csv", index=False)

    


 
     
     
     


if __name__ == "__main__":
    modelling = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "modelling_table.csv")
    build_classification_model(modelling)