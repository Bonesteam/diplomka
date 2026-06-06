"""Модуль для навчання baseline ML моделей для порівняння з нейронними мережами."""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import joblib, os

BASELINES = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1),
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42, learning_rate=0.1, max_depth=5),
}

def train_baselines(X_train, y_train, X_test, y_test, save_dir="saved_models"):
    """
    Тренує всі baseline ML моделі та зберігає результати.
    
    Args:
        X_train, y_train: Тренувальні дані та мітки
        X_test, y_test: Тестові дані та мітки
        save_dir: Директорія для збереження моделей
    
    Returns:
        dict: Метрики для кожної моделі (accuracy, f1_weighted)
    
    Моделі включають RandomForest, SVM та GradientBoosting з балансуванням класів.
    """
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
