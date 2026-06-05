"""
generate_dataset.py
-------------------
Розширення оригінального датасету Plant Health Biosensor Dataset
(Kaggle, 1 254 зразки) до будь-якої точної кількості зразків (наприклад, 15 438)
за допомогою параметричного синтезу на основі урізаного нормального розподілу.

Алгоритм:
  1. Завантажуємо оригінальний датасет.
  2. Обчислюємо реальні статистики (mean, std, min, max) для кожного
     класу та кожної ознаки — безпосередньо з оригінальних даних.
  3. Визначаємо пропорції класів в оригінальному датасеті.
  4. Динамічно розраховуємо кількість синтетичних зразків на кожен клас,
     щоб фінальний датасет мав ТОЧНУ кількість рядків (наприклад, 15 438)
     та зберігав оригінальний баланс класів.
  5. Генеруємо синтетичні зразки через TruncatedNormal(mean, std, min, max).
  6. Додаємо прикордонні зразки (Hard Samples) на межах суміжних класів.
  7. Об'єднуємо оригінальні + синтетичні + прикордонні зразки та зберігаємо результат.

Запуск:
  python generate_dataset.py --total 15438
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

# ========================== ПАРАМЕТРИ ЗА ЗАМОВЧУВАННЯМ ==========================
DEFAULT_ORIGINAL = "plant_health_biosensor_dataset.csv"
DEFAULT_OUTPUT   = "data/plant_health_biosensor_15k.csv"
DEFAULT_TOTAL    = 15438
RANDOM_SEED      = 42

TARGET_COL = "plant_health_status"
FEATURE_COLS = [
    "fluorescence_intensity", "colorimetric_index", "spr_signal_strength",
    "leaf_temperature", "chlorophyll_content", "moisture_level",
    "light_absorption_ratio", "volatile_organic_compounds",
]

# ========================== ОБЧИСЛЕННЯ СТАТИСТИК ==========================

def compute_class_stats(df: pd.DataFrame) -> dict:
    """
    Обчислює mean / std / min / max для кожної ознаки та кожного класу
    безпосередньо з реальних даних.
    """
    stats = {}
    for cls in sorted(df[TARGET_COL].unique()):
        subset = df[df[TARGET_COL] == cls][FEATURE_COLS]
        stats[int(cls)] = {}
        for col in FEATURE_COLS:
            stats[int(cls)][col] = {
                "mean": float(subset[col].mean()),
                "std":  float(subset[col].std()),
                "min":  float(subset[col].min()),
                "max":  float(subset[col].max()),
            }
    return stats


# ========================== ГЕНЕРАЦІЯ РЯДКІВ ==========================

def generate_row(class_label: int, class_stats: dict) -> dict:
    """Генерує один синтетичний рядок для заданого класу."""
    row = {}
    for col in FEATURE_COLS:
        s = class_stats[class_label][col]
        # Захист від нульового std
        std = s["std"] if s["std"] > 1e-6 else 1e-6
        a = (s["min"] - s["mean"]) / std
        b = (s["max"] - s["mean"]) / std
        value = truncnorm.rvs(a, b, loc=s["mean"], scale=std)
        value = np.clip(value, s["min"], s["max"])
        # Зберігаємо оригінальну точність без округлення
        row[col] = float(value)
    return row


def generate_synthetic(class_stats: dict, synthetic_per_class: dict) -> pd.DataFrame:
    """Генерує синтетичні зразки відповідно до заданого розподілу."""
    all_rows = []
    print("Генерація синтетичних зразкiв...")
    for cls, count in sorted(synthetic_per_class.items()):
        print(f"  Клас {cls}: {count} зразкiв")
        for _ in range(count):
            row = generate_row(cls, class_stats)
            row[TARGET_COL] = cls
            all_rows.append(row)
    return pd.DataFrame(all_rows)


# ========================== ПРИКОРДОННІ ЗРАЗКИ ==========================

def add_hard_samples(class_stats: dict, n_samples: int = 800) -> pd.DataFrame:
    """
    Додає прикордонні зразки на межах суміжних класів.
    Значення генеруються як рівномірна інтерполяція між центроїдами двох сусідніх класів.
    Гарантує повернення точної кількості n_samples зразків.
    """
    if n_samples <= 0:
        return pd.DataFrame(columns=FEATURE_COLS + [TARGET_COL])
        
    hard_rows = []
    boundary_pairs = [(2, 3), (1, 2), (0, 1)]  # межі класів

    for i in range(n_samples):
        cls_a, cls_b = boundary_pairs[i % len(boundary_pairs)]
        row = {}
        for col in FEATURE_COLS:
            mean_a = class_stats[cls_a][col]["mean"]
            mean_b = class_stats[cls_b][col]["mean"]
            std_a  = class_stats[cls_a][col]["std"]
            # Точка між двома центроїдами + невеликий шум
            alpha = np.random.uniform(0.35, 0.65)
            val = alpha * mean_a + (1 - alpha) * mean_b
            val += np.random.normal(0, std_a * 0.15)
            # Межі — об'єднання обох класів
            lo = min(class_stats[cls_a][col]["min"], class_stats[cls_b][col]["min"])
            hi = max(class_stats[cls_a][col]["max"], class_stats[cls_b][col]["max"])
            val = np.clip(val, lo, hi)
            # Зберігаємо оригінальну точність без округлення
            row[col] = float(val)
        row[TARGET_COL] = np.random.choice([cls_a, cls_b])
        hard_rows.append(row)

    return pd.DataFrame(hard_rows)


# ========================== ВАЛІДАЦІЯ ==========================

def validate(df: pd.DataFrame):
    print("\n" + "=" * 52)
    print("ВАЛІДАЦIЯ ДАТАСЕТУ")
    print("=" * 52)

    print("\nРозподiл класiв у фінальному датасеті:")
    total = len(df)
    for cls in sorted(df[TARGET_COL].unique()):
        cnt = (df[TARGET_COL] == cls).sum()
        print(f"  Клас {cls}: {cnt:>5}  ({cnt / total * 100:.1f}%)")

    print("\nСереднi значення по класах (ключовi ознаки):")
    for cls in sorted(df[TARGET_COL].unique()):
        sub = df[df[TARGET_COL] == cls]
        fi  = sub["fluorescence_intensity"]
        ci  = sub["colorimetric_index"]
        spr = sub["spr_signal_strength"]
        print(f"  Клас {cls}:  fluor={fi.mean():.1f}+-{fi.std():.1f}  "
              f"color={ci.mean():.3f}+-{ci.std():.3f}  "
              f"SPR={spr.mean():.0f}+-{spr.std():.0f}")

    # Перевірка розділення між крайніми класами
    m0 = df[df[TARGET_COL] == 0]["fluorescence_intensity"].mean()
    m3 = df[df[TARGET_COL] == 3]["fluorescence_intensity"].mean()
    print(f"\nРiзниця мiж кл.0 i кл.3 (fluor): {m0 - m3:.1f}")


# ========================== ГОЛОВНА ФУНКЦІЯ ==========================

def main(original_path: str, output_path: str, total: int):
    np.random.seed(RANDOM_SEED)
    
    print("=" * 52)
    print("РОЗШИРЕННЯ ДАТАСЕТУ PLANT HEALTH BIOSENSOR")
    print("=" * 52)

    # 1. Завантаження оригінального датасету
    if not os.path.exists(original_path):
        raise FileNotFoundError(
            f"Оригiнальний датасет не знайдено: {original_path}\n"
            "Завантажте його з Kaggle (programmer3/plant-health-biosensor-dataset)"
        )

    df_orig = pd.read_csv(original_path)
    df_orig[TARGET_COL] = df_orig[TARGET_COL].astype(int)
    df_orig = df_orig[FEATURE_COLS + [TARGET_COL]]  # залишаємо лише потрібні колонки

    n_orig = len(df_orig)
    print(f"\nОригiнальний датасет: {n_orig} рядкiв")
    
    print("\nРозподіл оригінальних класів:")
    class_counts = df_orig[TARGET_COL].value_counts().to_dict()
    for cls in sorted(class_counts.keys()):
        cnt = class_counts[cls]
        print(f"  Клас {cls}: {cnt} рядкiв ({cnt / n_orig * 100:.1f}%)")

    # 2. Обчислення реальних статистик з оригінальних даних
    class_stats = compute_class_stats(df_orig)

    # 3. Визначення кількості синтетичних зразків для кожного класу
    # Фінальний обсяг: total = n_synth + n_hard (оригінальні не додаємо, щоб уникнути витоку в тест)
    n_hard = 800
    n_synth = total - n_hard
    
    if n_synth < 0:
        n_synth = 0
        n_hard = total
        
    print(f"\nЦільовий обсяг датасету: {total} рядків")
    print(f"  - Реальні зразки: 0 (збережено окремо як незалежний тест)")
    print(f"  - Прикордонні зразки (Hard): {n_hard}")
    print(f"  - Необхідно згенерувати синтетичних: {n_synth}")

    # Розрахунок пропорцій та розподіл синтетичних зразків
    synthetic_per_class = {}
    current_sum = 0
    # Сортуємо класи за спаданням частоти для правильного розподілу залишків
    sorted_classes = sorted(class_counts.keys(), key=lambda c: class_counts[c], reverse=True)
    
    for i, cls in enumerate(sorted_classes):
        prop = class_counts[cls] / n_orig
        if i == len(sorted_classes) - 1:
            # Останній клас забирає залишок, щоб сума була точно n_synth
            count = n_synth - current_sum
        else:
            count = int(round(n_synth * prop))
            current_sum += count
        synthetic_per_class[cls] = max(0, count)

    print("\nРозподіл генерації по класах:")
    for cls, count in sorted(synthetic_per_class.items()):
        print(f"  Клас {cls}: буде згенеровано {count} зразків")

    # 4. Генерація синтетичних зразків на основі реальних статистик
    df_synth = generate_synthetic(class_stats, synthetic_per_class)

    # 5. Додавання прикордонних зразків (Hard)
    df_hard = add_hard_samples(class_stats, n_samples=n_hard)

    # 6. Об'єднання: тільки синтетичні + прикордонні (для незалежності тесту)
    df_final = pd.concat([df_synth, df_hard], ignore_index=True)

    # 7. Перемішування
    df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Забезпечуємо точний зріз у випадку будь-яких неочікуваних відхилень
    if len(df_final) != total:
        print(f"\n[Попередження] Довжина отриманого датасету {len(df_final)} відрізняється від цільової {total}. Виконуємо коригування...")
        if len(df_final) > total:
            df_final = df_final.iloc[:total].reset_index(drop=True)
        else:
            # Якщо раптом менше, добираємо випадковими дублікатами згенерованих
            extra = total - len(df_final)
            df_extra = df_synth.sample(n=extra, replace=True, random_state=RANDOM_SEED)
            df_final = pd.concat([df_final, df_extra], ignore_index=True)
            df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # 8. Валідація
    validate(df_final)

    # 9. Збереження
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_final.to_csv(output_path, index=False)

    print(f"\nЗбережено у: {output_path}")
    print(f"Загальний фінальний розмір: {len(df_final)} рядкiв (точно як замовлено!)")


# ========================== ТОЧКА ВХОДУ ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Розширення датасету Plant Health Biosensor до точного обсягу"
    )
    parser.add_argument(
        "--original", default=DEFAULT_ORIGINAL,
        help=f"Шлях до оригiнального CSV (default: {DEFAULT_ORIGINAL})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Шлях для збереження результату (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--total", type=int, default=DEFAULT_TOTAL,
        help=f"Цiльова кiлькiсть рядкiв (default: {DEFAULT_TOTAL})"
    )
    args = parser.parse_args()
    main(args.original, args.output, args.total)
