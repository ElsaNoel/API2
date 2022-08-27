from flask import Flask, request, jsonify
# Add needed package
import pickle
# import numpy as np
import pyforest
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
import shap

app = Flask(__name__)

# Read raw dataset 
with open('raw_data.pkl', 'rb') as inp:
    df = pd.DataFrame(pickle.load(inp))

# Read pretrained model
with open('lGBM_model.pkl', 'rb') as inp:
    lgbm = pickle.load(inp)

# Read pretrained model
with open('scaler.pkl', 'rb') as inp:
    ss = pickle.load(inp)

# Load explainer
with open('shap_explainer_lGBM.pkl', 'rb') as inp:
    explainer = pickle.load(inp)

df_ss = pd.DataFrame(ss.transform(df.drop(columns='TARGET')), columns = df.drop(columns='TARGET').columns)
df_ss.SK_ID_CURR = list(df.SK_ID_CURR)

sk_id_curr = df_ss.SK_ID_CURR

@app.route("/")
def get_indexpage():
    return "Page d'accueil"

# Define endpoint that return the dataset
@app.route("/raw_dataset/")
def get_dataset():
    return df.to_json() # Pour passer des trucs via les endpoints, on doit utiliser le format JSON !

# Define endpoint that return the dataset
@app.route("/scaled_dataset/")
def get_scaled_dataset():
    return df_ss.to_json() 
@app.route("/get_idx/")
def get_idx():
    return sk_id_curr.to_json()

@app.route('/predict/', methods = ['GET'])
def get_predict():
    args = request.args
    id_ = args.get("id", type=int)

    idx = np.where(df_ss.SK_ID_CURR == id_)[0][0]

    prediction = lgbm.predict_proba(df_ss.drop(columns='SK_ID_CURR').iloc[[idx]])[0][1]
    


    return jsonify(str(prediction))

if __name__ == "__main__":
    app.run(debug=True)