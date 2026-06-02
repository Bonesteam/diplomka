import joblib, os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_scaler(t="standard"):
    return MinMaxScaler() if t == "minmax" else StandardScaler()

def fit_transform(X_train, scaler_type="standard"):
    sc = get_scaler(scaler_type)
    return sc.fit_transform(X_train), sc

def transform(X, scaler):
    if X is None:
        return None
    return scaler.transform(X)

def save_scaler(scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)
