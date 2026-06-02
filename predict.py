import os, sys, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SCALER_PATH = "saved_models/scaler.pkl"
MODEL_PATH  = "saved_models/mlp_best.keras"

FEATURE_COLS = ["fluorescence_intensity","colorimetric_index","spr_signal_strength",
                "leaf_temperature","chlorophyll_content","moisture_level",
                "light_absorption_ratio","volatile_organic_compounds"]
FEATURE_UA = ["Інтенсивність флуоресценції","Колориметричний індекс","SPR-сигнал",
              "Температура листа (°C)","Вміст хлорофілу","Рівень вологості",
              "Коефіцієнт поглинання світла","Леткі органічні сполуки (VOC)"]
FEATURE_RANGES = [(15,90,"норма: 45–60"),(0.25,0.95,"норма: 0.55–0.70"),
                  (40,180,"норма: 85–115"),(9,41,"норма: 20–30"),
                  (10,72,"норма: 33–50"),(13,48,"норма: 25–35"),
                  (0.50,0.87,"норма: 0.65–0.74"),(0,29,"норма: 10–19")]
CLASS_NAMES  = {0:"Критичний стрес",1:"Помірний стрес",2:"Легкий стрес",3:"Здорова рослина"}
STATUS_COLOR = {0:"\033[91m",1:"\033[93m",2:"\033[94m",3:"\033[92m"}
RESET="\033[0m"; BOLD="\033[1m"

# Реальні центроїди класів з датасету
PRESETS = {
    "1": {"name":"Критичний стрес",  "values":[61.2,0.60,100.3,24.3,52.0,30.3,0.71,14.9]},
    "2": {"name":"Помірний стрес",   "values":[50.0,0.61,124.0,24.6,34.6,30.2,0.70,15.4]},
    "3": {"name":"Легкий стрес",     "values":[50.9,0.44, 91.0,24.1,42.3,24.4,0.71,15.4]},
    "4": {"name":"Здорова рослина",  "values":[49.0,0.61, 92.4,24.8,39.6,30.0,0.70,14.8]},
}

def check_models():
    for p in [MODEL_PATH, SCALER_PATH]:
        if not os.path.exists(p):
            print(f"\n[!] Не знайдено: {p}\n    Спочатку: запустіть графічний інтерфейс: python app.py → вкладка 'Навчання'\n"); sys.exit(1)

def input_float(prompt, lo, hi):
    while True:
        try:
            val = float(input(f"  {prompt}: "))
            if lo <= val <= hi: return val
            print(f"    Діапазон {lo}–{hi}")
        except ValueError: print("    Введіть число")
        except KeyboardInterrupt: print("\nВихід."); sys.exit(0)

def enter_manually():
    print(f"\n{BOLD}Введіть дані біосенсорів:{RESET}")
    values = []
    for feat, (lo, hi, hint) in zip(FEATURE_UA, FEATURE_RANGES):
        print(f"\n  {feat}  [{hint}]")
        values.append(input_float(f"Діапазон {lo}–{hi}", lo, hi))
    return values

def choose_preset():
    print(f"\n{BOLD}Оберіть шаблон:{RESET}")
    for k, v in PRESETS.items(): print(f"  {k}. {v['name']}")
    while True:
        try:
            ch = input("\n  Вибір (1–4): ").strip()
            if ch in PRESETS:
                vals = PRESETS[ch]["values"]
                print(f"\n  Завантажено: {PRESETS[ch]['name']}")
                for i, feat in enumerate(FEATURE_UA): print(f"    {feat}: {vals[i]}")
                return vals
            print("  Введіть 1–4")
        except KeyboardInterrupt: print("\nВихід."); sys.exit(0)

def show_result(values, probs):
    cls = int(np.argmax(probs)); color = STATUS_COLOR[cls]; name = CLASS_NAMES[cls]
    print(f"\n{'-'*58}\n{BOLD}  Результат:{RESET}")
    print(f"  Стан:       {color}{BOLD}{name}{RESET}")
    print(f"  Впевненість:{color}{BOLD} {probs[cls]*100:.1f}%{RESET}")
    print(f"\n  {BOLD}Ймовірності:{RESET}")
    bar_w = 28
    for i, (n, p) in enumerate(zip(CLASS_NAMES.values(), probs)):
        filled = int(p*100/100*bar_w)
        bar = "█"*filled + "░"*(bar_w-filled)
        marker = " ◄" if i == cls else ""
        print(f"  {STATUS_COLOR[i]}{bar}{RESET}  {p*100:5.1f}%  {n}{marker}")
    print(f"\n  {BOLD}Показники:{RESET}")
    for i, (feat, (lo, hi, hint), val) in enumerate(zip(FEATURE_UA, FEATURE_RANGES, values)):
        ok = lo <= val <= hi
        st = f"\033[92m[ok]\033[0m" if ok else f"\033[91m[!] {hint}\033[0m"
        print(f"    {feat:<40} {val:>8.3f}  {st}")
    print(f"{'-'*58}")

def run_from_file(csv_path, scaler, model):
    import pandas as pd
    from models.predictor import predict
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y_true = df["plant_health_status"].values.astype(int) if "plant_health_status" in df.columns else None
    X_sc = scaler.transform(X); y_pred, y_proba = predict(model, X_sc)
    print(f"\nЗнайдено {len(X)} зразків. Перші 10:\n")
    print(f"  {'#':>4}  {'Передбачено':<20}  {'Впевн.':>7}", end="")
    if y_true is not None: print(f"  {'Справжній':<20}  {'Вірно?':>6}")
    else: print()
    correct = 0
    for i in range(min(10, len(X))):
        cls = y_pred[i]; conf = y_proba[i][cls]*100; name = CLASS_NAMES[cls]
        line = f"  {i+1:>4}  {STATUS_COLOR[cls]}{name:<20}{chr(27)}[0m  {conf:>6.1f}%"
        if y_true is not None:
            ok = y_pred[i] == y_true[i]
            ok_c = "\033[92m" if ok else "\033[91m"
            line += f"  {CLASS_NAMES[y_true[i]]:<20}  {ok_c}{'+'if ok else '-'}\033[0m"
            if ok: correct += 1
        print(line)
    if y_true is not None:
        from sklearn.metrics import accuracy_score
        print(f"\n  Точність (весь файл): {accuracy_score(y_true, y_pred)*100:.2f}%")

def main():
    print(f"\n{'='*58}\n{BOLD}   Система оцінювання стану здоров'я рослин{RESET}\n{'='*58}")
    check_models()
    from preprocessing.scaler import load_scaler
    from models.predictor import load_model, predict
    scaler = load_scaler(SCALER_PATH); model = load_model(MODEL_PATH)

    while True:
        print(f"\n{BOLD}Що зробити?{RESET}\n  1. Ввести дані вручну\n  2. Готовий шаблон\n  3. Завантажити CSV\n  0. Вийти")
        try: choice = input("\n  Вибір: ").strip()
        except KeyboardInterrupt: print("\nВихід."); break
        if choice == "0": print("\nДо побачення!\n"); break
        elif choice == "1":
            values = enter_manually()
            X_sc = scaler.transform(np.array([values], dtype=np.float32))
            _, probs = predict(model, X_sc); show_result(values, probs[0])
        elif choice == "2":
            values = choose_preset()
            X_sc = scaler.transform(np.array([values], dtype=np.float32))
            _, probs = predict(model, X_sc); show_result(values, probs[0])
        elif choice == "3":
            try:
                path = input("\n  Шлях до CSV: ").strip().strip('"')
                if os.path.exists(path): run_from_file(path, scaler, model)
                else: print(f"  Файл не знайдено: {path}")
            except Exception as e: print(f"  Помилка: {e}")
        try:
            again = input("\n  Ще раз? (Enter = так, n = ні): ").strip().lower()
            if again == "n": print("\nДо побачення!\n"); break
        except KeyboardInterrupt: print("\nВихід."); break

if __name__ == "__main__":
    main()
