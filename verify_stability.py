"""
Перевірка стабільності класифікації: малі зміни входу не повинні різко міняти клас.
Запуск: python verify_stability.py
"""
import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from analysis_logic import classify, calibrated_confidence
from preprocessing.scaler import load_scaler
from models.predictor import load_model

# Копія діапазонів з app.py
NORMAL_RANGES = [
    (45, 60, ""), (0.55, 0.70, ""), (85, 115, ""), (20, 30, ""),
    (33, 50, ""), (25, 35, ""), (0.65, 0.74, ""), (10, 19, ""),
]
PRESETS = {
    "Критичний": [30.0, 0.30, 40.0, 35.0, 20.0, 15.0, 0.50, 5.0],
    "Помірний": [50.2, 0.61, 124.93, 24.68, 34.15, 30.10, 0.70, 15.43],
    "Легкий": [50.9, 0.46, 94.73, 24.12, 41.04, 25.13, 0.70, 15.34],
    "Здоровий": [48.91, 0.61, 92.46, 24.82, 39.62, 29.96, 0.70, 14.78],
}
CLASS_NAMES = ["Критичний", "Помірний", "Легкий", "Здоровий"]
EXPECTED = {"Критичний": 0, "Помірний": 1, "Легкий": 2, "Здоровий": 3}


def predict_raw(model, scaler, values):
    X = scaler.transform(np.array([values], dtype=np.float32))
    p = model.predict(X, verbose=0)[0]
    return p / p.sum()


def run_checks(model, scaler):
    ok = True
    print("=== Шаблони (очікуваний клас) ===")
    for name, vals in PRESETS.items():
        p = predict_raw(model, scaler, vals)
        r = classify(vals, p, NORMAL_RANGES)
        exp = EXPECTED[name]
        got = r["final_cls"]
        mark = "OK" if got == exp else "FAIL"
        if got != exp:
            ok = False
        print(
            f"  [{mark}] {name}: {CLASS_NAMES[got]} "
            f"conf={r['confidence']:.1f}% "
            f"(відхилень {r['n_clear_deviations']})"
        )

    print("\n=== Стабільність: +5% до одного показника (здоровий шаблон) ===")
    base = PRESETS["Здоровий"].copy()
    p0 = predict_raw(model, scaler, base)
    c0 = classify(base, p0, NORMAL_RANGES)["final_cls"]
    flips = 0
    for i in range(8):
        v = base.copy()
        delta = abs(v[i]) * 0.05 + 0.05
        v[i] += delta
        r = classify(v, predict_raw(model, scaler, v), NORMAL_RANGES)
        if r["final_cls"] != c0:
            flips += 1
            print(
                f"  [!] ознака {i}: клас {CLASS_NAMES[c0]} -> {CLASS_NAMES[r['final_cls']]} "
                f"(conf {r['confidence']:.1f}%)"
            )
    if flips == 0:
        print("  OK — клас не змінився при невеликих змінах")
    else:
        print(f"  Увага: {flips}/8 ознак змінили клас при +5%")
        if flips > 3:
            ok = False

    print("\n=== Межа колориметрії (легкий шаблон, поступово) ===")
    mild = PRESETS["Легкий"].copy()
    prev = None
    for val in [0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.55, 0.56, 0.58]:
        v = mild.copy()
        v[1] = val
        r = classify(v, predict_raw(model, scaler, v), NORMAL_RANGES)
        cls = r["final_cls"]
        line = f"  color={val:.2f} -> {CLASS_NAMES[cls]} conf={r['confidence']:.1f}%"
        if prev is not None and cls != prev:
            line += "  (зміна)"
        print(line)
        prev = cls

    print("\n=== Впевненість не завжди >95% ===")
    high = 0
    for name, vals in PRESETS.items():
        r = classify(vals, predict_raw(model, scaler, vals), NORMAL_RANGES)
        if r["confidence"] >= 95:
            high += 1
    if high == len(PRESETS):
        print("  [!] Усі шаблони >95% — можливо ще занадто різко")
    else:
        print(f"  OK — лише {high}/{len(PRESETS)} шаблонів з conf>=95%")

    print("\n" + ("УСПІХ" if ok else "Є ЗАУВАЖЕННЯ — перегляньте вивід вище"))
    return 0 if ok else 1


def main():
    if not os.path.exists("saved_models/mlp_best.keras"):
        print("Спочатку навчіть модель (вкладка «Навчання»).")
        return 1
    scaler = load_scaler("saved_models/scaler.pkl")
    model = load_model("saved_models/mlp_best.keras")
    return run_checks(model, scaler)


if __name__ == "__main__":
    sys.exit(main())
