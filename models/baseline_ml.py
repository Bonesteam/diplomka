from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import joblib, os

BASELINES = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
}

def train_baselines(X_train, y_train, X_test, y_test, save_dir="saved_models"):
    results = {}
    for name, clf in BASELINES.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        results[name] = {"accuracy": acc, "f1_weighted": f1}
        print(f"  {name:<22} acc={acc:.4f}  f1={f1:.4f}")
        joblib.dump(clf, os.path.join(save_dir, f"{name.lower()}.pkl"))
    return results
