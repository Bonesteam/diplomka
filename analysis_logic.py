"""
Стабільна логіка аналізу: м'які межі норм, пом'якшені ймовірності моделі, злиття з правилами.
"""
import numpy as np

# Допуск біля межі норми (частка ширини діапазону з кожного боку)
NORM_MARGIN_FRAC = 0.08

# Температура softmax — >1 зменшує надмірну впевненість моделі
MODEL_TEMPERATURE = 2.8

# Максимальна відображувана впевненість у UI
MAX_DISPLAY_CONFIDENCE = 88.0


def _range_width(lo, hi):
    return max(hi - lo, 1e-9)


def feature_deviation_status(val, lo, hi):
    """
    Статус показника:
      'ok' — у нормі
      'borderline' — біля межі (не рахується як сильне відхилення)
      'deviation' — явно поза нормою
    """
    if lo <= val <= hi:
        return "ok", 0.0

    width = _range_width(lo, hi)
    margin = width * NORM_MARGIN_FRAC
    lo_ext, hi_ext = lo - margin, hi + margin

    if lo_ext <= val <= hi_ext:
        # трохи за межами «строгої» норми, але в зоні допуску
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
    """Повертає список (status, severity) для кожної ознаки."""
    out = []
    for val, (lo, hi, _) in zip(values, normal_ranges):
        out.append(feature_deviation_status(val, lo, hi))
    return out


def count_clear_deviations(statuses):
    return sum(1 for s, _ in statuses if s == "deviation")


def total_stress_score(statuses):
    return sum(sev for _, sev in statuses)


def soften_model_probs(probs, temperature=MODEL_TEMPERATURE):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0)
    p = p / p.sum()
    logits = np.log(p)
    e = np.exp(logits / temperature)
    return e / e.sum()


def rule_probs_from_stress(n_clear, stress_total):
    """Ймовірності за показниками; при масових відхиленнях — висока впевненість у критичному."""
    s = np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float64)

    if n_clear >= 7 or stress_total >= 6.0:
        s[:] = [0.93, 0.04, 0.02, 0.01]
    elif n_clear >= 5 or stress_total >= 4.5:
        s[:] = [0.86, 0.08, 0.04, 0.02]
    elif n_clear >= 4 or stress_total >= 3.2:
        s[:] = [0.68, 0.18, 0.10, 0.04]
    elif n_clear >= 3 or stress_total >= 2.2:
        s[:] = [0.48, 0.28, 0.16, 0.08]
    elif n_clear == 2 or stress_total >= 1.3:
        s[:] = [0.12, 0.48, 0.28, 0.12]
    elif n_clear == 1 or stress_total >= 0.45:
        s[:] = [0.06, 0.22, 0.46, 0.26]
    else:
        s[:] = [0.04, 0.08, 0.18, 0.70]

    return s / s.sum()


def blend_weight(n_clear, stress_total, model_probs_raw=None):
    """
    Частка правил у фінальному рішенні.
    При багатьох відхиленнях модель часто помиляється (зокрема на 999…) — покладаємось на показники.
    """
    if n_clear == 0 and stress_total < 0.35:
        return 0.0
    if n_clear >= 7 or stress_total >= 6.0:
        return 1.0
    if n_clear >= 5 or stress_total >= 4.5:
        return 0.95
    if n_clear >= 4:
        return 0.82
    # модель суперечить показникам (типово «здорова» при багатьох червоних)
    if model_probs_raw is not None and n_clear >= 3:
        p = np.asarray(model_probs_raw, dtype=np.float64)
        p = p / p.sum()
        if int(np.argmax(p)) == 3 and p[3] > 0.5:
            return 0.95
    w = n_clear * 0.11 + stress_total * 0.07
    return float(np.clip(w, 0.0, 0.75))


def calibrated_confidence(probs, n_clear=0, blend_w=0.0):
    """Впевненість для UI; не занижувати, коли рішення явно за показниками."""
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    top = float(p.max())
    second = float(np.partition(p, -2)[-2])
    margin = top - second
    conf = top * 100.0

    # багато відхилень + домінують правила — довіряємо топ-ймовірності
    if blend_w >= 0.9 and n_clear >= 5:
        return min(conf, MAX_DISPLAY_CONFIDENCE)

    if margin < 0.22:
        conf = 48.0 + margin * 145.0
    return min(conf, MAX_DISPLAY_CONFIDENCE)


def classify(values, model_probs, normal_ranges):
    """
    Повертає dict:
      final_probs, final_cls, confidence, statuses,
      model_probs_soft, rule_probs, blend_w, model_cls
    """
    statuses = analyze_features(values, normal_ranges)
    n_clear = count_clear_deviations(statuses)
    stress = total_stress_score(statuses)

    raw_p = np.asarray(model_probs, dtype=np.float64)
    raw_p = raw_p / raw_p.sum()
    model_soft = soften_model_probs(raw_p)
    model_cls = int(np.argmax(model_soft))
    rule_p = rule_probs_from_stress(n_clear, stress)
    w = blend_weight(n_clear, stress, model_probs_raw=raw_p)

    if w >= 0.999:
        final_p = rule_p
    elif w <= 0.0:
        final_p = model_soft
    else:
        final_p = (1.0 - w) * model_soft + w * rule_p
        final_p = final_p / final_p.sum()

    final_cls = int(np.argmax(final_p))
    conf = calibrated_confidence(final_p, n_clear=n_clear, blend_w=w)

    return {
        "final_probs": final_p,
        "final_cls": final_cls,
        "confidence": conf,
        "statuses": statuses,
        "model_probs_soft": model_soft,
        "rule_probs": rule_p,
        "blend_w": w,
        "model_cls": model_cls,
        "n_clear_deviations": n_clear,
        "stress_total": stress,
    }
