import pandas as pd
import numpy as np

FEATURE_COLS = [
    "fluorescence_intensity", "colorimetric_index", "spr_signal_strength",
    "leaf_temperature", "chlorophyll_content", "moisture_level",
    "light_absorption_ratio", "volatile_organic_compounds",
]
# default logical target column name (internal)
TARGET_COL = "plant_health_status"

CLASS_NAMES = {
    0: "Критичний стрес",
    1: "Помірний стрес",
    2: "Легкий стрес",
    3: "Здорова рослина",
}

def _find_target_column(df):
    # try common variants case-insensitively
    candidates = [c for c in df.columns]
    lower_map = {c.lower(): c for c in candidates}
    # common variants
    for key in ("plant_health_status", "plant_health", "plantstatus", "plant_health_status"): # repeated safe
        if key in lower_map:
            return lower_map[key]
    # try words-based match
    for c in candidates:
        kl = c.lower().replace(' ', '_')
        if 'plant' in kl and 'health' in kl and 'status' in kl:
            return c
    # fallback: try any column containing 'health'
    for c in candidates:
        if 'health' in c.lower():
            return c
    return None


def load_data(path, n_features=None):
    df = pd.read_csv(path)
    target_col = _find_target_column(df)
    if target_col is None:
        raise KeyError(f"Could not find target column in {path}")
    # if target values are strings like 'High Stress'/'Healthy', map to integer labels
    from pandas.api import types as ptypes
    if ptypes.is_string_dtype(df[target_col].dtype) or not ptypes.is_integer_dtype(df[target_col].dtype):
        codes, uniques = pd.factorize(df[target_col])
        df[target_col] = codes
    df[target_col] = df[target_col].astype(int)

    # find feature columns in the CSV that best match expected FEATURE_COLS
    def _match_feature(col_name, candidates):
        c = col_name.lower()
        for tok in candidates:
            if tok in c:
                return True
        return False

    selected_features = []
    for fname in FEATURE_COLS:
        tokens = fname.split('_')
        found = None
        for col in df.columns:
            if col == target_col:
                continue
            if all(tok in col.lower() for tok in tokens):
                found = col
                break
        if found is None:
            # try partial match
            for col in df.columns:
                if col == target_col:
                    continue
                if any(tok in col.lower() for tok in tokens):
                    found = col
                    break
        if found is None:
            # fallback: try to select numeric columns automatically
            numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target_col]
            # exclude obvious meta columns
            exclude = set(["timestamp", "plant_id", "id"])
            numeric_cols = [c for c in numeric_cols if c.lower() not in exclude]
            if len(numeric_cols) > 0:
                # use numeric columns as features
                selected_features = numeric_cols
                break
            else:
                raise KeyError(f"Could not find feature column matching '{fname}' in {path} and no numeric fallback available")
        selected_features.append(found)

    # adjust to requested number of features if provided
    if n_features is not None:
        if len(selected_features) > n_features:
            selected_features = selected_features[:n_features]
        elif len(selected_features) < n_features:
            # add other numeric columns not yet used until we reach n_features
            extras = [c for c in df.select_dtypes(include=["number"]).columns if c not in selected_features and c != target_col]
            for c in extras:
                selected_features.append(c)
                if len(selected_features) >= n_features:
                    break
            if len(selected_features) < n_features:
                raise KeyError(f"Not enough numeric columns to reach requested n_features={n_features}")

    X = df[selected_features].values.astype(np.float32)
    y = df[target_col].values.astype(np.int32)
    return X, y, df
