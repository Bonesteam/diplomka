import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_auc_score, confusion_matrix
from preprocessing.loader import CLASS_NAMES

def evaluate(y_true, y_pred, y_proba=None):
    """
    Розширена оцінка якості моделі
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_micro = float(f1_score(y_true, y_pred, average="micro"))
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics = {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
    }

    # per-class metrics
    try:
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    except Exception:
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []

    # confusion matrix and support
    cm = confusion_matrix(y_true, y_pred)
    support = cm.sum(axis=1).tolist()

    metrics.update({
        "precision_per_class": [float(x) for x in precision_per_class],
        "recall_per_class": [float(x) for x in recall_per_class],
        "f1_per_class": [float(x) for x in f1_per_class],
        "confusion_matrix": cm.tolist(),
        "support": support,
        "class_names": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())],
    })
    
    # pretty print summary
    print("\n" + "="*50)
    print("  МЕТРИКИ ЯКОСТІ МОДЕЛІ")
    print("="*50)
    for k in ["accuracy", "f1_weighted", "f1_macro", "precision_weighted", "recall_weighted"]:
        v = metrics.get(k, 0.0)
        print(f"  {k:20s}: {v:.4f}")

    print("\n  МАТРИЦЯ ПОМИЛОК:")
    print("  " + "-" * 50)
    class_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    print("     " + "".join(f"{name[:10]:>10}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"  {class_names[i][:10]:10s}" + "".join(f"{x:10d}" for x in row))

    print("\n  ДЕТАЛЬНИЙ ЗВІТ ПО КЛАСАХ:")
    target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    # ROC-AUC overall and per-class if probabilities provided
    if y_proba is not None:
        try:
            roc_auc_overall = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
            roc_auc_per_class = roc_auc_score(y_true, y_proba, multi_class="ovr", average=None)
            metrics["roc_auc"] = roc_auc_overall
            metrics["roc_auc_per_class"] = [float(x) for x in roc_auc_per_class]
            print(f"\n  roc_auc: {roc_auc_overall:.4f}")
        except Exception as e:
            print(f"  ROC-AUC не обчислено: {e}")
            metrics["roc_auc"] = 0.0

    return metrics