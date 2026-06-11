"""Гібридна класифікація: MLP (78%) + відповідність профілю класу за 8 показниками (22%)."""
import numpy as np

NORM_MARGIN_FRAC = 0.20
LOW_CONFIDENCE = 65.0
CLOSE_MARGIN = 0.15
HEALTHY_CLASS = 3
MODEL_BLEND_WEIGHT = 0.78  # частка нейромережі; решта — профілі 8 біосенсорів

DEFAULT_NORMAL_RANGES = [
    (39.25, 58.65, "профіль здорової: 39–59"),
    (0.52, 0.70, "профіль здорової: 0.52–0.70"),
    (75.38, 109.49, "профіль здорової: 75–109"),
    (19.94, 29.73, "профіль здорової: 20–30"),
    (29.65, 49.55, "профіль здорової: 30–50"),
    (24.99, 35.03, "профіль здорової: 25–35"),
    (0.65, 0.75, "профіль здорової: 0.65–0.75"),
    (9.97, 19.59, "профіль здорової: 10–20"),
]

DEFAULT_INPUT_RANGES = [
    (15, 90, "допустимо: 15–90"),
    (0.25, 0.95, "допустимо: 0.25–0.95"),
    (40, 180, "допустимо: 40–180"),
    (9, 41, "допустимо: 9–41"),
    (10, 72, "допустимо: 10–72"),
    (13, 48, "допустимо: 13–48"),
    (0.50, 0.87, "допустимо: 0.50–0.87"),
    (0, 29, "допустимо: 0–29"),
]


def _fmt_val(v):
    if abs(v) >= 100 or (abs(v) < 0.01 and v != 0):
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _range_hint(lo, hi, prefix="профіль здорової"):
    return f"{prefix}: {_fmt_val(lo)}–{_fmt_val(hi)}"


def compute_normal_ranges(df, feature_cols, target_col, reference_class=HEALTHY_CLASS,
                          lo_pct=10, hi_pct=90):
    """Діапазони на базі середнього ± 0.8 стд (здорового класу) — охоплює ~57% здорових рослин."""
    sub = df[df[target_col] == reference_class]
    if len(sub) < 10:
        return list(DEFAULT_NORMAL_RANGES)
    ranges = []
    for col in feature_cols:
        mean = float(sub[col].mean())
        std = float(sub[col].std())
        # Діапазон: середня ± 0.8*стд — охоплює ~57% здорових рослин
        # Достатньо розсунутий, щоб здорові рослини показували "норму",
        # але чітко виявляє значущі відхилення (стрес)
        lo = mean - 0.8 * std
        hi = mean + 0.8 * std
        if hi <= lo:
            hi = lo + 1e-6
        ranges.append((lo, hi, _range_hint(lo, hi)))
    return ranges


def compute_input_ranges(df, feature_cols, lo_pct=1, hi_pct=99):
    """Широкі межі для полів введення — за всім датасетом."""
    ranges = []
    for col in feature_cols:
        lo = float(df[col].quantile(lo_pct / 100.0))
        hi = float(df[col].quantile(hi_pct / 100.0))
        if hi <= lo:
            hi = lo + 1e-6
        ranges.append((lo, hi, f"допустимо: {_fmt_val(lo)}–{_fmt_val(hi)}"))
    return ranges


def _range_width(lo, hi):
    return max(hi - lo, 1e-9)


def feature_deviation_status(val, lo, hi):
    if lo <= val <= hi:
        return "ok", 0.0

    width = _range_width(lo, hi)
    margin = width * NORM_MARGIN_FRAC
    lo_ext, hi_ext = lo - margin, hi + margin

    if lo_ext <= val <= hi_ext:
        if val < lo:
            frac = (lo - val) / margin
        else:
            frac = (val - hi) / margin
        severity = 0.15 + 0.25 * min(1.0, frac)
        return "borderline", severity

    if val < lo:
        severity = min(1.0, (lo - margin - val) / width)
    else:
        severity = min(1.0, (val - hi - margin) / width)
    return "deviation", max(0.45, severity)


def analyze_features(values, normal_ranges):
    """Повертає список (status, severity) для кожної ознаки — лише для UI."""
    out = []
    for val, (lo, hi, _) in zip(values, normal_ranges):
        out.append(feature_deviation_status(val, lo, hi))
    return out


def count_clear_deviations(statuses):
    return sum(1 for s, _ in statuses if s == "deviation")


def compute_class_profiles(df, feature_cols, target_col, n_classes=4, regularization_factor=0.15):
    """
    Середнє та σ кожного класу — для злиття з показниками.
    Застосовує згладжування дисперсії (Variance Smoothing / Shrinkage)
    для запобігання надмірної чутливості до дрібних змін.
    """
    means = np.zeros((n_classes, len(feature_cols)), dtype=np.float64)
    stds = np.zeros((n_classes, len(feature_cols)), dtype=np.float64)
    
    global_stds = df[feature_cols].std().values
    
    for c in range(n_classes):
        sub = df[df[target_col] == c][feature_cols]
        class_means = sub.mean().values
        class_stds = sub.std().values
        
        # Обмежуємо стандартне відхилення знизу як відсоток від глобального стандартного відхилення
        smoothed_stds = np.maximum(class_stds, regularization_factor * global_stds)
        
        means[c] = class_means
        stds[c] = np.maximum(smoothed_stds, 1e-6)
    return means, stds


def _feature_status_weight(status, severity):
    """Вага ознаки: червоний > жовтий > зелений, але всі 8 враховуються."""
    if status == "ok":
        return 0.15
    if status == "borderline":
        return 0.45 + 0.35 * severity
    return 0.85 + 0.15 * severity


def compute_feature_evidence_probs(values, statuses, class_means, class_stds):
    """
    Ймовірності класів за відповідністю профілю.
    Кожен з 8 показників впливає з вагою за статусом (ok/borderline/deviation).
    """
    vals = np.asarray(values, dtype=np.float64)
    n_classes = class_means.shape[0]
    log_scores = np.zeros(n_classes, dtype=np.float64)

    for i, (status, severity) in enumerate(statuses):
        w = _feature_status_weight(status, severity)
        for c in range(n_classes):
            z = (vals[i] - class_means[c, i]) / class_stds[c, i]
            log_scores[c] -= w * 0.5 * (z ** 2)

    log_scores -= log_scores.max()
    exp = np.exp(log_scores)
    return exp / exp.sum()


def model_confidence(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    return float(p.max() * 100.0)


def classify(values, model_probs, normal_ranges, class_means=None, class_stds=None,
             model_weight=MODEL_BLEND_WEIGHT):
    statuses = analyze_features(values, normal_ranges)
    n_dev = count_clear_deviations(statuses)

    p_model = np.asarray(model_probs, dtype=np.float64)
    p_model = p_model / p_model.sum()
    model_cls = int(np.argmax(p_model))

    if class_means is not None and class_stds is not None:
        p_feat = compute_feature_evidence_probs(values, statuses, class_means, class_stds)
        p = model_weight * p_model + (1.0 - model_weight) * p_feat
        p = p / p.sum()
    else:
        p = p_model.copy()

    final_cls = int(np.argmax(p))
    return {
        "final_probs": p,
        "final_cls": final_cls,
        "confidence": model_confidence(p),
        "statuses": statuses,
        "n_deviations": n_dev,
        "model_cls": model_cls,
        "feature_influence": 1.0 - model_weight if class_means is not None else 0.0,
    }


def prediction_caution(res, statuses, class_names):
    """Попередження для пограничних або суперечливих результатів."""
    p = np.asarray(res["final_probs"], dtype=np.float64)
    p = p / p.sum()
    final_cls = res["final_cls"]
    conf = res["confidence"]
    n_dev = res.get("n_deviations", count_clear_deviations(statuses))
    order = np.argsort(p)
    second_cls = int(order[-2])
    second_p = float(p[second_cls])
    margin = float(p[final_cls] - second_p)

    lines = []
    
    # Попередження про невпевненість моделі
    if conf < LOW_CONFIDENCE:
        lines.append(
            f"Модель не впевнена ({conf:.0f}%) — другий варіант: "
            f"«{class_names[second_cls]}» ({second_p * 100:.0f}%)."
        )
    elif margin < CLOSE_MARGIN:
        lines.append(
            f"Класи близькі: «{class_names[final_cls]}» ({conf:.0f}%) "
            f"проти «{class_names[second_cls]}» ({second_p * 100:.0f}%)."
        )
    
    model_cls = res.get("model_cls", final_cls)

    # Попередження про конфлікти між моделлю та показниками
    if n_dev >= 4 and model_cls == HEALTHY_CLASS and final_cls == HEALTHY_CLASS:
        lines.append(
            f"⚠ КОНФЛІКТ: {n_dev} показників сильно відхиляються, "
            f"але модель передбачає здорову рослину. Слід розглянути як легкий стрес."
        )
    elif n_dev >= 6 and final_cls != 0:  # Не критичний стрес (клас 0)
        lines.append(
            f"⚠ КОНФЛІКТ: {n_dev}/8 показників сильно відхиляються, "
            f"але модель не вказує на критичний стрес."
        )
    elif n_dev == 0 and final_cls != HEALTHY_CLASS:
        lines.append(
            f"✓ Всі показники в нормі (профіль здорової рослини), "
            f"але модель передбачає {class_names[final_cls].lower()}. Можливо помилка."
        )

    if not lines:
        return None
    return {
        "title": "Пограничний випадок — інтерпретуйте обережно",
        "lines": lines,
    }