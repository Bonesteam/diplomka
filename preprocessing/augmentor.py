from imblearn.over_sampling import SMOTE
import numpy as np

def apply_smote(X_train, y_train, random_state=42):
    counts = np.bincount(y_train)
    print(f"До SMOTE: {dict(enumerate(counts))}")
    sm = SMOTE(random_state=random_state, k_neighbors=min(5, min(counts)-1))
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"Після SMOTE: {dict(enumerate(np.bincount(y_res)))}")
    return X_res, y_res