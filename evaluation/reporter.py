import json, os
from datetime import datetime
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
