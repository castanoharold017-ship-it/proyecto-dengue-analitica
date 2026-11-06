
import pandas as pd, numpy as np, json, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

ROOT=os.path.dirname(os.path.abspath(__file__))
df=pd.read_csv(os.path.join(ROOT,"data","dengue_limpio.csv"))
target="GROUPS: DENGUE FEVER OR COMPLICATED DENGUE"
features=["PCV_GE_42","PLATELETS_LE_75000","HIGHEST_WBC_GE_7000","ALT_GE_70","BP_CODE_2"]
X=df[features].astype(float); y=(df[target].astype(int)==2).astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
models={
 "logistic":Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("scaler",StandardScaler()),("model",LogisticRegression(class_weight="balanced",max_iter=2000,random_state=42))]),
 "neural":Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("scaler",StandardScaler()),("model",MLPClassifier(hidden_layer_sizes=(16,8),activation="relu",solver="adam",alpha=0.01,max_iter=1500,early_stopping=True,random_state=42))])
}
metrics={}
os.makedirs(os.path.join(ROOT,"models"),exist_ok=True)
for name,model in models.items():
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    prob=model.predict_proba(X_test)[:,1]
    metrics[name]={
      "accuracy":round(float(accuracy_score(y_test,pred)),4),
      "precision":round(float(precision_score(y_test,pred,zero_division=0)),4),
      "recall":round(float(recall_score(y_test,pred,zero_division=0)),4),
      "f1":round(float(f1_score(y_test,pred,zero_division=0)),4),
      "auc":round(float(roc_auc_score(y_test,prob)),4),
      "confusion_matrix":confusion_matrix(y_test,pred).tolist()
    }
    joblib.dump(model,os.path.join(ROOT,"models",name+".joblib"))
with open(os.path.join(ROOT,"models","metrics.json"),"w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)
print(json.dumps({"rows":len(df),"class_counts":y.value_counts().sort_index().to_dict(),"metrics":metrics},indent=2))
