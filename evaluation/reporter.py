import json, os
from datetime import datetime
import csv
import numpy as np


def _make_json_safe(obj):
    """Recursively convert objects to JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _make_json_safe(obj.tolist())
    try:
        return float(obj)
    except Exception:
        try:
            return int(obj)
        except Exception:
            return str(obj)

def save_report(metrics, cv_results, baseline_results, save_path="results/report.json", extra=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    report = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "neural_network": metrics, "cross_validation": cv_results, "baselines": baseline_results}
    if extra:
        report.update(extra)
    # If an ablation CSV exists in the results folder, try to include it
    ablation_csv = os.path.join(os.path.dirname(save_path), "ablation_table.csv")
    if os.path.exists(ablation_csv):
        try:
            experiments = []
            with open(ablation_csv, newline="", encoding="utf-8") as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    # Normalize keys if they differ
                    name = row.get('Конфігурація') or row.get('name') or row.get('configuration')
                    acc = row.get('Accuracy') or row.get('accuracy')
                    f1w = row.get('F1 (weighted)') or row.get('f1_weighted')
                    f1m = row.get('F1 (macro)') or row.get('f1_macro')
                    roc = row.get('ROC-AUC') or row.get('roc_auc')
                    experiments.append({
                        "name": name,
                        "accuracy": float(acc) if acc not in (None, "") else None,
                        "f1_weighted": float(f1w) if f1w not in (None, "") else None,
                        "f1_macro": float(f1m) if f1m not in (None, "") else None,
                        "roc_auc": float(roc) if roc not in (None, "") else None,
                    })
            report.setdefault("ablation", {})
            report["ablation"]["experiments"] = experiments
            report["ablation"]["csv"] = os.path.relpath(ablation_csv).replace('\\', '/')
        except Exception:
            pass
    # remove training_history if present (user requested it not be in report.json)
    if "training_history" in report:
        try:
            del report["training_history"]
        except Exception:
            pass
    # sanitize report to ensure JSON serializability
    safe_report = _make_json_safe(report)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(safe_report, f, ensure_ascii=False, indent=2)
    print(f"Звіт збережено: {save_path}")

def print_comparison(metrics, baseline_results):
    print("\n=== Порівняння методів ===")
    print(f"{'Метод':<25} {'Accuracy':>10} {'F1 (weighted)':>15}")
    print("-" * 52)
    print(f"{'MLP (нейромережа)':<25} {metrics['accuracy']:>10.4f} {metrics['f1_weighted']:>15.4f}")
    for name, res in baseline_results.items():
        print(f"{name:<25} {res['accuracy']:>10.4f} {res['f1_weighted']:>15.4f}")
