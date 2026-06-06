"""
АРХІТЕКТУРА КЛАСИФІКАЦІЇ:

Система двоходова класифікації для діагностики стану рослини:

1. МОДЕЛЬ (Нейронна мережа MLP)
   - Входи: 8 біосенсорних показників (нормалізовані)
   - Виходи: 4 ймовірності для класів (softmax + temperature scaling)
   - Структура: [128 -> 64 -> 32] нейронів з ReLU
   - Training: SMOTE для балансування, class_weights для рідких класів
   - Prediction: temperature scaling T=1.3 для реалістичних ймовірностей

2. АНАЛІЗ ПОКАЗНИКІВ (Feature-based analysis)
   - Порівнює кожен показник з профілем здорової рослини (± 0.5σ)
   - Класифікує статус: "ok" (в нормі), "borderline" (на межі), "deviation" (відхилення)
   - Рахує кількість відхилень (0-8)
   - Це чисто описовий аналіз, не впливає на результат моделі напряму

3. FEATURE BOOSTING (направлений)
   - Посилює КОНКРЕТНИЙ стресовий клас, який нейромережа вже вважає провідним
   - Вирішує два парадокси:
     * "5 червоних прапорців, але модель каже здорова" — пригнічує p_healthy
     * "критичний і легкий стрес мають однакову кількість прапорців" —
       посилення йде лише туди, куди вказує модель (не рівномірно по всіх стресах)
   - Посилення залежить від n_deviations:
     * 0-1: довіряємо моделі повністю
     * 2-3: буст провідного стресового класу ×1.3–1.6
     * 4-5: буст ×1.8–2.2
     * 6+:  буст ×2.8; якщо провідний — легкий стрес, перенаправляємо на критичний

4. СИСТЕМА ПОПЕРЕДЖЕНЬ
   - Показує конфлікти між моделлю та показниками
   - Пояснює коли результат невпевнений або парадоксальний

РЕЗУЛЬТАТ: Комбінована оцінка яка враховує і нейронну мережу, і біосенсорні показники
"""
import numpy as np

NORM_MARGIN_FRAC = 0.20
LOW_CONFIDENCE = 65.0
CLOSE_MARGIN = 0.15
HEALTHY_CLASS = 3

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


def model_confidence(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    return float(p.max() * 100.0)


def apply_feature_boosting(model_probs, n_deviations):
    """
    Направлене посилення стресового класу на основі кількості відхилень показників.

    Ключова відмінність від рівномірного boosting:
    Посилюється КОНКРЕТНИЙ стресовий клас, який нейромережа вже вважає
    найімовірнішим — а не всі стресові класи одразу. Це усуває парадокс:
    "легкий стрес і критичний стрес мають однакову кількість прапорців,
    але однакове посилення" — тепер посилення йде туди, куди вказує модель.

    Схема:
    - 0-1 відхилень : довіряємо моделі повністю
    - 2-3 відхилень : помірний буст провідного стресового класу (×1.3–1.6)
    - 4-5 відхилень : сильний буст провідного стресового класу (×1.8–2.2)
    - 6+  відхилень : критичний буст (×2.8); якщо провідний клас — легкий стрес,
                      він «підвищується» до критичного (захист від недооцінки)
    У всіх випадках p_healthy зменшується пропорційно до n_deviations.
    """
    p = np.asarray(model_probs, dtype=np.float64)
    p = p / p.sum()

    if n_deviations <= 1:
        return p

    STRESS_CLASSES = [0, 1, 2]  # легкий, помірний, критичний

    # Знаходимо провідний стресовий клас (той, якому модель дала найбільшу p серед стресових)
    leading_stress = max(STRESS_CLASSES, key=lambda c: p[c])

    # Шкала буст-коефіцієнтів і штраф для "здорова" залежно від n_deviations
    #   (boost_factor, healthy_penalty)
    BOOST_TABLE = {
        2: (1.30, 0.80),
        3: (1.60, 0.60),
        4: (1.80, 0.40),
        5: (2.20, 0.20),
    }
    if n_deviations <= 5:
        boost_factor, healthy_penalty = BOOST_TABLE[n_deviations]
    else:  # 6+
        boost_factor, healthy_penalty = 2.80, 0.10

    p_boosted = p.copy()

    # --- Спеціальний випадок для 6+ відхилень ---
    # Якщо провідний стресовий клас — легкий стрес (0), але відхилень дуже багато,
    # модель, ймовірно, недооцінює серйозність. Перенаправляємо буст на критичний (2).
    if n_deviations >= 6 and leading_stress == 0:
        leading_stress = 2  # критичний стрес

    # Посилюємо ЛИШЕ провідний стресовий клас
    p_boosted[leading_stress] *= boost_factor

    # Знижуємо "здорова"
    p_boosted[HEALTHY_CLASS] *= healthy_penalty

    # Нормалізація — сума = 1
    p_boosted = p_boosted / p_boosted.sum()

    return p_boosted


def classify(values, model_probs, normal_ranges):
    statuses = analyze_features(values, normal_ranges)
    n_dev = count_clear_deviations(statuses)
    
    # Застосовуємо boosting на основі кількості девіацій
    p = apply_feature_boosting(model_probs, n_dev)
    
    final_cls = int(np.argmax(p))
    return {
        "final_probs": p,
        "final_cls": final_cls,
        "confidence": model_confidence(p),
        "statuses": statuses,
        "n_deviations": n_dev,  # Для аналізу
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
    
    # Попередження про конфлікти між моделлю та показниками
    if n_dev >= 4 and final_cls == HEALTHY_CLASS:
        lines.append(
            f"⚠ КОНФЛІКТ: {n_dev} показників сильно відхиляються, "
            f"але модель передбачає здорову рослину. Слід розглянути як легкий стрес."
        )
    elif n_dev >= 6 and final_cls != 2:  # Не критичний стрес
        lines.append(
            f"⚠ КОНФЛІКТ: {n_dev}/8 показників вказують на критичний стрес, "
            f"але модель обережна. Посилено критичність."
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