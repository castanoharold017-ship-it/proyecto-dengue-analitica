
import os, json, joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

BASE=os.path.dirname(os.path.abspath(__file__))
app=Flask(__name__)
FEATURES=["PCV_GE_42","PLATELETS_LE_75000","HIGHEST_WBC_GE_7000","ALT_GE_70","BP_CODE_2"]
MODELS={
 "logistic":joblib.load(os.path.join(BASE,"models","logistic.joblib")),
 "neural":joblib.load(os.path.join(BASE,"models","neural.joblib"))
}
with open(os.path.join(BASE,"models","metrics.json"),encoding="utf-8") as f:
    METRICS=json.load(f)

@app.get("/")
def index():
    return render_template("index.html", metrics=METRICS)

@app.post("/predict")
def predict():
    try:
        pcv=float(request.form.get("pcv",""))
        platelets=float(request.form.get("platelets",""))
        wbc=float(request.form.get("wbc",""))
        alt=float(request.form.get("alt",""))
        bp_code=int(request.form.get("bp_code",""))
        model_name=request.form.get("model","logistic")
        if model_name not in MODELS: raise ValueError("Modelo no válido.")
        if bp_code not in (1,2): raise ValueError("El código de presión debe ser 1 o 2.")
        if min(pcv,platelets,wbc,alt)<0: raise ValueError("Los valores no pueden ser negativos.")
        row=np.array([[
            int(pcv>=42), int(platelets<=75000), int(wbc>=7000),
            int(alt>=70), int(bp_code==2)
        ]],dtype=float)
        model=MODELS[model_name]
        probability=float(model.predict_proba(row)[0,1])
        prediction=int(model.predict(row)[0])
        return jsonify({
            "ok":True,
            "prediction":prediction,
            "label":"Mayor probabilidad de dengue complicado" if prediction==1 else "Mayor probabilidad de dengue no complicado",
            "probability":round(probability*100,1),
            "model":"Regresión logística" if model_name=="logistic" else "Red neuronal artificial",
            "indicators":{
                "PCV ≥ 42%":bool(pcv>=42),
                "Plaquetas ≤ 75.000":bool(platelets<=75000),
                "WBC ≥ 7.000":bool(wbc>=7000),
                "ALT ≥ 70 U/L":bool(alt>=70),
                "Código de presión = 2":bool(bp_code==2)
            }
        })
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),400

@app.get("/health")
def health(): return {"status":"ok"}

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)),debug=False)
