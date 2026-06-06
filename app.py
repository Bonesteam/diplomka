import os, sys, threading, numpy as np, json, datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATA_PATH = "plant_health_biosensor_dataset.csv"

FEATURE_COLS = ["fluorescence_intensity","colorimetric_index","spr_signal_strength",
                "leaf_temperature","chlorophyll_content","moisture_level",
                "light_absorption_ratio","volatile_organic_compounds"]
FEATURE_UA = ["Флуоресценція","Колориметричний індекс","SPR-сигнал","Температура листа (°C)",
              "Вміст хлорофілу","Рівень вологості","Поглинання світла","Леткі сполуки (VOC)"]

CLASS_NAMES  = ["Критичний стрес","Помірний стрес","Легкий стрес","Здорова рослина"]
CLASS_COLORS = ["#E24B4A","#EF9F27","#639922","#1D9E75"]
CLASS_BG     = ["#FCEBEB","#FAEEDA","#EAF3DE","#E1F5EE"]

# Резервні пресети — реальні зразки з датасету, що найкраще представляють кожен клас.
# Відібрані як найбільш типові (мінімальна відстань до центроїду класу)
# і максимально контрастні відносно інших класів.
#
# Клас 0 — Критичний стрес: висока флуоресценція (>60), хлорофіл >50 (парадоксально
#   підвищений через стресову реакцію), SPR ~100
# Клас 1 — Помірний стрес: SPR значно підвищений (>120), хлорофіл знижений (<37)
# Клас 2 — Легкий стрес: низький колориметр (<0.52), низька вологість (<22), знижений SPR
# Клас 3 — Здорова рослина: збалансовані показники, всі в нормальних діапазонах
FALLBACK_PRESETS = {
    "Критичний стрес": [61.49, 0.64, 104.49, 24.08, 57.07, 32.04, 0.71, 13.23],
    "Помірний стрес":  [52.24, 0.64, 120.54, 27.62, 36.49, 30.17, 0.74, 16.57],
    "Легкий стрес":    [52.03, 0.50, 89.48, 24.47, 43.60, 20.33, 0.74, 14.51],
    "Здорова рослина": [53.03, 0.63, 95.28, 22.63, 40.44, 30.39, 0.71, 13.65],
}

ACCENT="#534AB7"; ACCENT2="#7F77DD"; BG="#F8F8F8"; CARD="#FFFFFF"; BORDER="#E0DED8"; TEXT="#1A1A1A"; MUTED="#6B6B68"
WARN_BG="#FAEEDA"; WARN_FG="#854F0B"

from analysis_logic import (
    classify, prediction_caution,
    compute_normal_ranges, compute_input_ranges,
    DEFAULT_NORMAL_RANGES, DEFAULT_INPUT_RANGES,
)


class PlantHealthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Оцінювання стану здоров'я рослин")
        self.geometry("1080x720"); self.minsize(900,620); self.configure(bg=BG)
        self.model=None; self.scaler=None; self._model_loaded=False

        self.normal_ranges = list(DEFAULT_NORMAL_RANGES)
        self.feature_ranges = list(DEFAULT_INPUT_RANGES)
        self.presets = dict(FALLBACK_PRESETS)
        self.class_means = None
        self.class_stds = None
        self._load_dataset_context()
        self._build_ui(); self._try_load_model_silent()

    def _dataset_path(self):
        if os.path.exists(DATA_PATH):
            return DATA_PATH
        return None

    def _load_dataset_context(self):
        from preprocessing.loader import TARGET_COL
        import pandas as pd

        path = self._dataset_path()
        if path is None:
            return
        try:
            df = pd.read_csv(path)
            self.normal_ranges = compute_normal_ranges(df, FEATURE_COLS, TARGET_COL)
            self.feature_ranges = compute_input_ranges(df, FEATURE_COLS)
            grp = df.groupby(TARGET_COL)[FEATURE_COLS]
            means = grp.mean()
            stds = grp.std().replace(0, 1e-6)
            self.class_means = means.values
            self.class_stds = stds.values
            # Пресети — реальні зразки датасету, найближчі до центроїду класу
            # (типовіші за середнє, яке може опинитись між класами)
            presets = {}
            for i in range(len(CLASS_NAMES)):
                sub = df[df[TARGET_COL] == i][FEATURE_COLS]
                centroid = means.loc[i]
                dists = ((sub - centroid) ** 2).sum(axis=1)
                best_idx = dists.idxmin()
                presets[CLASS_NAMES[i]] = [round(float(v), 2) for v in df.loc[best_idx, FEATURE_COLS].tolist()]
            self.presets = presets
        except Exception:
            pass

    def _build_ui(self):
        hdr = tk.Frame(self, bg=ACCENT, height=54); hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="Система оцінювання стану здоров'я рослин",
                 bg=ACCENT, fg="white", font=("Segoe UI",14,"bold")).pack(side="left",padx=20,pady=12)
        self.lbl_model_status = tk.Label(hdr, text="● модель не завантажена",
                                         bg=ACCENT, fg="#FFCCCC", font=("Segoe UI",10))
        self.lbl_model_status.pack(side="right", padx=20)
        
        style = ttk.Style(self); style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG, foreground=MUTED, font=("Segoe UI",10), padding=[16,8])
        style.map("TNotebook.Tab", background=[("selected",CARD)], foreground=[("selected",ACCENT)])
        style.configure("TProgressbar", troughcolor=BORDER, background=ACCENT2, thickness=6)
        
        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True, padx=12, pady=(8,12))
        self.tab_predict = tk.Frame(nb, bg=BG)
        self.tab_train   = tk.Frame(nb, bg=BG)
        self.tab_batch   = tk.Frame(nb, bg=BG)
        self.tab_info    = tk.Frame(nb, bg=BG)
        
        nb.add(self.tab_predict, text="  Аналіз рослини  ")
        nb.add(self.tab_train,   text="  Навчання моделі  ")
        nb.add(self.tab_batch,   text="  Пакетний аналіз CSV  ")
        nb.add(self.tab_info,    text="  Довідка  ")
        
        self._build_predict_tab()
        self._build_train_tab()
        self._build_batch_tab()
        self._build_info_tab()

    # ==================== СТОРІНКА ДОВІДКА ====================
    def _build_info_tab(self):
        info = tk.Frame(self.tab_info, bg=BG)
        info.pack(fill="both", expand=True, padx=16, pady=12)
        
        # Заголовок
        tk.Label(info, text="📊 Про систему", bg=BG, fg=ACCENT, 
                 font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Опис
        desc = tk.Label(info, text="Система оцінює стан здоров'я рослин на основі 8 біосенсорних показників. "
                        "Використовується нейронна мережа MLP з точністю ~76%.", 
                        bg=BG, fg=TEXT, font=("Segoe UI", 11), wraplength=750, justify="left")
        desc.pack(anchor="w", pady=(0, 15))
        
        # Таблиця ознак
        tk.Label(info, text="🔬 Біосенсорні показники", bg=BG, fg=ACCENT, 
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))
        
        # Рамка з таблицею
        table_frame = tk.Frame(info, bg=CARD, relief="flat", bd=1, highlightbackground=BORDER, highlightthickness=1)
        table_frame.pack(fill="x", pady=(0, 15))
        
        cols = ("Ознака", "Нормальний діапазон", "Одиниці вимірювання")
        for i, col in enumerate(cols):
            lbl = tk.Label(table_frame, text=col, bg=ACCENT, fg="white", font=("Segoe UI", 10, "bold"),
                           padx=10, pady=6)
            lbl.grid(row=0, column=i, sticky="ew")
        
        data = [
            ("Флуоресценція", "39–59", "відносні од."),
            ("Колориметричний індекс", "0.52–0.70", "одиниці"),
            ("SPR-сигнал", "75–109", "нм"),
            ("Температура листа", "20–30", "°C"),
            ("Вміст хлорофілу", "30–50", "мг/г"),
            ("Рівень вологості", "25–35", "%"),
            ("Поглинання світла", "0.65–0.75", "коеф."),
            ("Леткі сполуки (VOC)", "10–20", "ppm")
        ]
        
        for i, row_data in enumerate(data):
            for j, val in enumerate(row_data):
                bg_color = "#F8F9FA" if i % 2 == 0 else "#FFFFFF"
                lbl = tk.Label(table_frame, text=val, bg=bg_color, fg=TEXT, font=("Segoe UI", 10),
                               padx=10, pady=4)
                lbl.grid(row=i+1, column=j, sticky="ew")
        
        for j in range(3):
            table_frame.columnconfigure(j, weight=1)
        
        # Про модель
        tk.Label(info, text="🧠 Про нейронну мережу", bg=BG, fg=ACCENT, 
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))
        
        model_frame = tk.Frame(info, bg=CARD, relief="flat", bd=1, highlightbackground=BORDER, highlightthickness=1)
        model_frame.pack(fill="x", pady=(0, 15))
        
        model_info = [
            "• Архітектура: MLP (багатошарова нейронна мережа)",
            "• Розмір шарів: 128 → 64 → 32 нейрони",
            "• Функція активації: ReLU",
            "• Регуляризація: Dropout 0.3, BatchNormalization",
            "• Оптимізатор: Adam (learning rate = 0.001)",
            "• Функція втрат: sparse_categorical_crossentropy",
            "• Точність на тестових даних: ~76%",
            "• ROC-AUC (макро): ~0.94"
        ]
        
        for i, line in enumerate(model_info):
            tk.Label(model_frame, text=line, bg=CARD, fg=TEXT, font=("Segoe UI", 10),
                     anchor="w", padx=12, pady=3).pack(fill="x")
        
        # Інструкція
        tk.Label(info, text="📖 Як користуватися", bg=BG, fg=ACCENT, 
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))
        
        inst_frame = tk.Frame(info, bg=CARD, relief="flat", bd=1, highlightbackground=BORDER, highlightthickness=1)
        inst_frame.pack(fill="x")
        
        instructions = [
            "1. Введіть або завантажте показники біосенсорів на вкладці «Аналіз рослини»",
            "2. Натисніть «Аналізувати» для отримання результату",
            "3. Система покаже ймовірний стан рослини та впевненість",
            "4. Всі аналізи зберігаються в «Історія аналізів»",
            "5. Для масового аналізу використовуйте «Пакетний аналіз CSV»"
        ]
        
        for i, line in enumerate(instructions):
            tk.Label(inst_frame, text=line, bg=CARD, fg=TEXT, font=("Segoe UI", 10),
                     anchor="w", padx=12, pady=3).pack(fill="x")

    # ==================== ІНШІ МЕТОДИ (БЕЗ ЗМІН) ====================
    def _build_predict_tab(self):
        p = self.tab_predict
        left = tk.Frame(p, bg=BG, width=420); left.pack(side="left", fill="y", padx=(0,6)); left.pack_propagate(False)
        right = tk.Frame(p, bg=BG); right.pack(side="left", fill="both", expand=True)
        
        card0 = self._card(left, "Швидкі шаблони")
        for name in self.presets:
            tk.Button(card0, text=name, bg=BG, fg=ACCENT, font=("Segoe UI",9), relief="flat",
                      bd=0, cursor="hand2", activeforeground=ACCENT2,
                      command=lambda n=name: self._load_preset(n)).pack(anchor="w", pady=1)
        
        card1 = self._card(left, "Дані біосенсорів"); self.entries = []
        defaults = self.presets.get("Здорова рослина", list(self.presets.values())[0])
        for i, (ua, (lo, hi, hint)) in enumerate(zip(FEATURE_UA, self.feature_ranges)):
            row = tk.Frame(card1, bg=CARD); row.pack(fill="x", pady=3)
            tk.Label(row, text=ua, bg=CARD, fg=TEXT, font=("Segoe UI",9,"bold"), anchor="w", width=28).pack(side="left")
            var = tk.StringVar(value=str(defaults[i]))
            e = tk.Entry(row, textvariable=var, width=9, font=("Segoe UI",10), relief="flat",
                         bg="#F0EFF8", fg=TEXT, highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=ACCENT2)
            e.pack(side="left", padx=(4,2))
            tk.Label(row, text=hint, bg=CARD, fg=MUTED, font=("Segoe UI",8)).pack(side="left", padx=4)
            self.entries.append(var)
        
        self.btn_analyze = tk.Button(left, text="  Аналізувати  ", bg=ACCENT, fg="white",
            font=("Segoe UI",11,"bold"), relief="flat", cursor="hand2",
            activebackground=ACCENT2, padx=10, pady=8, command=self._run_predict)
        self.btn_analyze.pack(fill="x", padx=8, pady=(10,0))
        self.result_frame = tk.Frame(right, bg=BG); self.result_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self._show_placeholder()

    def _show_placeholder(self):
        for w in self.result_frame.winfo_children(): w.destroy()
        tk.Label(self.result_frame, text="Введіть дані біосенсорів\nі натисніть «Аналізувати»",
                 bg=BG, fg=MUTED, font=("Segoe UI",13)).pack(expand=True)

    def _show_result(self, values, probs):
        for w in self.result_frame.winfo_children(): w.destroy()
        res = classify(values, probs, self.normal_ranges)
        final_cls = res["final_cls"]
        display_probs = res["final_probs"]
        conf = res["confidence"]
        statuses = res["statuses"]
        caution = prediction_caution(res, statuses, CLASS_NAMES)

        if caution:
            warn = tk.Frame(self.result_frame, bg=WARN_BG, bd=0)
            warn.pack(fill="x", pady=(0, 8))
            tk.Label(warn, text=f"⚠  {caution['title']}", bg=WARN_BG, fg=WARN_FG,
                     font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x", padx=12, pady=(8, 2))
            for line in caution["lines"]:
                tk.Label(warn, text=f"• {line}", bg=WARN_BG, fg=WARN_FG,
                         font=("Segoe UI", 9), anchor="w", justify="left").pack(fill="x", padx=12, pady=1)
            tk.Frame(warn, bg=WARN_BG, height=6).pack()

        color = CLASS_COLORS[final_cls]
        bg_c = CLASS_BG[final_cls]
        banner = tk.Frame(self.result_frame, bg=bg_c, bd=0); banner.pack(fill="x", pady=(0,10))
        tk.Label(banner, text=CLASS_NAMES[final_cls], bg=bg_c, fg=color, font=("Segoe UI",18,"bold")).pack(pady=(14,2))
        conf_txt = f"Впевненість: {conf:.1f}%"
        margin = float(display_probs[final_cls] - np.partition(display_probs, -2)[-2])
        if margin < 0.15:
            conf_txt += "  (близькі класи — змінюйте показники поступово)"
        tk.Label(banner, text=conf_txt, bg=bg_c, fg=color, font=("Segoe UI",11)).pack(pady=(0,6))
        
        # Показуємо відхилення від норми здорової рослини — лише як довідкова інформація
        n_dev = res.get("n_deviations", 0)
        n_border = sum(1 for st, _ in statuses if st == "borderline")
        if n_dev == 0 and n_border == 0:
            dev_txt = "Усі показники в межах норми здорової рослини"
        elif n_dev == 0:
            dev_txt = f"Відхилень від норми немає  ·  на межі: {n_border}/8"
        elif n_border == 0:
            dev_txt = f"Відхилень від норми здорової рослини: {n_dev}/8"
        else:
            dev_txt = f"Відхилень від норми: {n_dev}/8  ·  на межі: {n_border}/8"
        tk.Label(banner, text=dev_txt, bg=bg_c, fg=color, font=("Segoe UI",9)).pack(pady=(0,4))
        tk.Label(banner, text="(клас визначено нейромережею за патерном усіх показників)",
                 bg=bg_c, fg=color, font=("Segoe UI",8)).pack(pady=(0,14))

        card_prob = self._card(self.result_frame, "Ймовірності класів")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, display_probs)):
            pct = prob * 100; row = tk.Frame(card_prob, bg=CARD); row.pack(fill="x", pady=3)
            tk.Label(row, text=name, bg=CARD, fg=TEXT, font=("Segoe UI",9), width=18, anchor="w").pack(side="left")
            track = tk.Frame(row, bg=BORDER, height=12, width=220); track.pack(side="left", padx=6); track.pack_propagate(False)
            fill_w = int(pct/100*220)
            if fill_w > 0: tk.Frame(track, bg=CLASS_COLORS[i], height=12, width=fill_w).place(x=0, y=0)
            tk.Label(row, text=f"{pct:.1f}%", bg=CARD, fg=MUTED, font=("Segoe UI",9), width=6).pack(side="left")
            if i == final_cls: tk.Label(row, text="◄", bg=CARD, fg=color, font=("Segoe UI",9)).pack(side="left")

        card_sens = self._card(self.result_frame, "Показники біосенсорів (порівняння з еталоном здорової рослини)")
        status_ui = {
            "ok": ("#1D9E75", "✔ норма"),
            "borderline": ("#E6A800", "⚠ на межі"),
            "deviation": ("#E24B4A", "✘ відхилення"),
        }

        for i, (ua, (_, _, hint), val) in enumerate(zip(FEATURE_UA, self.normal_ranges, values)):
            st, _ = statuses[i]
            val_color, dt = status_ui[st]
            lo, hi, _ = self.normal_ranges[i]
            row = tk.Frame(card_sens, bg=CARD); row.pack(fill="x", pady=2)
            tk.Label(row, text=ua, bg=CARD, fg=TEXT, font=("Segoe UI",9), width=32, anchor="w").pack(side="left")
            tk.Label(row, text=f"{val:.3f}", bg=CARD, fg=val_color, font=("Segoe UI",9,"bold"), width=8).pack(side="left")
            tk.Label(row, text=dt, bg=CARD, fg=val_color, font=("Segoe UI",9), width=14).pack(side="left")
            expected = f"еталон: {lo:.2f}-{hi:.2f}"
            tk.Label(row, text=expected, bg=CARD, fg=MUTED, font=("Segoe UI",8)).pack(side="left", padx=4)

        summary_lines = []
        for ua, (st, _) in zip(FEATURE_UA, statuses):
            if st == "deviation":
                summary_lines.append(f"✘ {ua}")
            elif st == "borderline":
                summary_lines.append(f"⚠ {ua}")
        if not summary_lines:
            summary_lines.append("Усі показники в нормі.")

        summary_text = "\n".join(summary_lines)
        summ = tk.Label(self.result_frame, text=summary_text,
                       bg=BG, fg=TEXT, font=("Segoe UI",10), justify="left")
        summ.pack(padx=8, pady=(6,12), anchor="w")

        self._save_to_history(values, CLASS_NAMES[final_cls], conf, display_probs.tolist())

    def _save_to_history(self, values, class_name, confidence, probs):
        """Save prediction history to a JSON file."""
        try:
            history_file = "results/prediction_history.json"
            history_data = []
            
            # Load existing history if file exists
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
            
            # Add new prediction record
            record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "input_values": {feat: val for feat, val in zip(FEATURE_COLS, values)},
                "predicted_class": class_name,
                "confidence": confidence,
                "class_probabilities": {cn: float(p) for cn, p in zip(CLASS_NAMES, probs)}
            }
            history_data.append(record)
            
            # Save updated history
            os.makedirs("results", exist_ok=True)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # Silently fail - history is optional
            pass

    def _load_preset(self, name):
        for var, v in zip(self.entries, self.presets[name]):
            var.set(str(v))

    def _run_predict(self):
        if not self._model_loaded:
            messagebox.showwarning("Модель не завантажена","Спочатку навчіть модель.")
            return
        try:
            values = [float(v.get()) for v in self.entries]
        except ValueError:
            messagebox.showerror("Помилка","Перевірте значення — мають бути числами.")
            return
        X = np.array([values], dtype=np.float32)
        X_sc = self.scaler.transform(X)
        raw_probs = self.model.predict(X_sc, verbose=0)[0]
        
        # Застосовуємо temperature scaling для калібрування вероятностей
        # Temperature = 1.3 робить розподіл більш плавним і реалістичним
        temperature = 1.3
        log_probs = np.log(raw_probs + 1e-9) / temperature
        probs = np.exp(log_probs)
        probs = probs / probs.sum()
        
        self._show_result(values, probs)

    def _build_train_tab(self):
        t=self.tab_train
        top=tk.Frame(t,bg=BG); top.pack(fill="x",padx=12,pady=10)
        
        # Load default values from config.yaml
        import yaml
        cfg = {}
        if os.path.exists("config.yaml"):
            try:
                with open("config.yaml") as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception: pass
            
        train_path = cfg.get("data", {}).get("path", DATA_PATH)
        epochs = str(cfg.get("training", {}).get("epochs", 100))
        batch_size = str(cfg.get("training", {}).get("batch_size", 32))
        lr = str(cfg.get("training", {}).get("learning_rate", 0.0008))
        patience = str(cfg.get("training", {}).get("patience", 15))
        val_size = str(cfg.get("data", {}).get("val_size", 0.20))
        apply_smote = cfg.get("preprocessing", {}).get("apply_smote", True)

        card_data=self._card(top,"Датасет")
        row1=tk.Frame(card_data,bg=CARD); row1.pack(fill="x", pady=2)
        tk.Label(row1,text="CSV:",bg=CARD,fg=TEXT,font=("Segoe UI",9), width=12, anchor="w").pack(side="left")
        self.var_csv=tk.StringVar(value=train_path)
        tk.Entry(row1,textvariable=self.var_csv,width=52,font=("Segoe UI",9),relief="flat",
                 bg="#F0EFF8",highlightthickness=1,highlightbackground=BORDER).pack(side="left",padx=6)
        tk.Button(row1,text="Огляд…",bg=BG,fg=ACCENT,font=("Segoe UI",9),relief="flat",cursor="hand2",
                  command=self._browse_csv).pack(side="left")
        tk.Label(card_data,
                 text="Розбиття: stratified train/validation (80/20 за замовч.). "
                      "Валідаційні зразки не входять до навчання.",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8), wraplength=520, justify="left").pack(anchor="w", pady=(4, 0))

        card_hp=self._card(top,"Гіперпараметри"); hp_grid=tk.Frame(card_hp,bg=CARD); hp_grid.pack(fill="x")
        params=[
            ("Епохи:", epochs),
            ("Batch size:", batch_size),
            ("Learning rate:", lr),
            ("Patience:", patience),
            ("Val split:", val_size)
        ]
        self.hp_vars=[]
        for i, (label, default) in enumerate(params):
            row = i // 4
            col = i % 4
            tk.Label(hp_grid,text=label,bg=CARD,fg=MUTED,font=("Segoe UI",9)).grid(row=row,column=col*2,sticky="w",padx=(0 if col==0 else 16,4),pady=4)
            var=tk.StringVar(value=default)
            tk.Entry(hp_grid,textvariable=var,width=8,font=("Segoe UI",10),relief="flat",
                     bg="#F0EFF8",highlightthickness=1,highlightbackground=BORDER).grid(row=row,column=col*2+1,sticky="w",pady=4)
            self.hp_vars.append(var)
        
        # SMOTE чекбокс
        smote_row = tk.Frame(card_hp, bg=CARD); smote_row.pack(fill="x", pady=(8, 4), padx=(0, 0))
        self.var_smote = tk.BooleanVar(value=apply_smote)
        smote_cb = tk.Checkbutton(smote_row, text="✓ Застосувати SMOTE (синтетичне перевищення менш частих класів)",
                                  bg=CARD, fg=ACCENT, font=("Segoe UI", 9, "bold"), 
                                  variable=self.var_smote, selectcolor=CARD, activebackground=CARD,
                                  activeforeground=ACCENT)
        smote_cb.pack(anchor="w", padx=4)
        tk.Label(smote_row, text="  📊 SMOTE покращує баланс класів та згенерує графік порівняння",
                bg=CARD, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w", padx=24, pady=(0, 4))
            
        btn_row=tk.Frame(t,bg=BG); btn_row.pack(fill="x",padx=12)
        self.btn_train=tk.Button(btn_row,text="  Почати навчання  ",bg=ACCENT,fg="white",
            font=("Segoe UI",11,"bold"),relief="flat",cursor="hand2",activebackground=ACCENT2,
            padx=12,pady=8,command=self._run_training_thread)
        self.btn_train.pack(side="left")
        self.lbl_epoch=tk.Label(btn_row,text="",bg=BG,fg=MUTED,font=("Segoe UI",9)); self.lbl_epoch.pack(side="left",padx=16)
        self.train_progress=ttk.Progressbar(t,mode="indeterminate"); self.train_progress.pack(fill="x",padx=12,pady=(6,0))
        log_frame=self._card(t,"Лог навчання"); log_frame.pack(fill="both",expand=True,padx=12,pady=6)
        self.log=ScrolledText(log_frame,height=14,font=("Consolas",9),bg="#1E1E1E",fg="#D4D4D4",
                              insertbackground="white",relief="flat",state="disabled")
        self.log.pack(fill="both",expand=True)
        self.log.tag_config("green",foreground="#4EC9B0"); self.log.tag_config("yellow",foreground="#DCDCAA")
        self.log.tag_config("red",foreground="#F44747"); self.log.tag_config("bold",foreground="#FFFFFF",font=("Consolas",9,"bold"))

    def _browse_csv(self):
        path=filedialog.askopenfilename(filetypes=[("CSV файли","*.csv"),("Всі файли","*.*")])
        if path: self.var_csv.set(path)

    def _log(self, msg, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", msg+"\n", tag if tag else "")
        self.log.see("end"); self.log.configure(state="disabled")

    def _run_training_thread(self):
        self.btn_train.configure(state="disabled",text="Навчання…"); self.train_progress.start(12)
        self.log.configure(state="normal"); self.log.delete("1.0","end"); self.log.configure(state="disabled")
        threading.Thread(target=self._run_training,daemon=True).start()

    def _run_training(self):
        import yaml
        try:
            self._log("="*50,"bold"); self._log("  Запуск навчання нейромережі","bold"); self._log("="*50,"bold")
            csv_src=self.var_csv.get().strip()
            if not os.path.exists(csv_src): self._log(f"[!] Файл не знайдено: {csv_src}","red"); return
            self._log(f"Датасет: {csv_src}","green")
            from preprocessing.loader import load_data
            from preprocessing.splitter import split_train_val
            from preprocessing.scaler import fit_transform,transform,save_scaler
            from models.mlp_model import build_mlp
            from models.cnn_model import build_cnn
            from models.trainer import train_model
            from models.baseline_ml import train_baselines
            from evaluation.metrics import evaluate
            from evaluation.confusion import plot_confusion_matrix
            from evaluation.reporter import save_report
            from evaluation.error_analysis import analyze_errors
            from evaluation.arch_comparison import compare_architectures
            from evaluation.cross_val import cross_validate_mlp
            from visualization.plots import plot_training_history,plot_class_distribution,plot_comparison_bar,plot_smote_comparison
            from visualization.feature_imp import plot_correlation_heatmap
            from visualization.roc_curves import plot_roc_curves
            ablation_results = []
            with open("config.yaml") as f: config=yaml.safe_load(f)
            config["training"]["epochs"]=int(self.hp_vars[0].get())
            config["training"]["batch_size"]=int(self.hp_vars[1].get())
            config["training"]["learning_rate"]=float(self.hp_vars[2].get())
            config["training"]["patience"]=int(self.hp_vars[3].get())
            config["data"]["val_size"]=float(self.hp_vars[4].get())
            config["data"]["path"]=csv_src
            config["preprocessing"]["apply_smote"]=self.var_smote.get()
            with open("config.yaml", "w") as f:
                yaml.safe_dump(config, f)
            os.makedirs("results",exist_ok=True); os.makedirs("saved_models",exist_ok=True)
            self._log("\n[1/8] Завантаження даних...","yellow")
            X,y,_=load_data(csv_src)
            self._log(f"  Зразків: {X.shape[0]}, ознак: {X.shape[1]}")
            try:
                plot_class_distribution(y, "Розподіл класів (оригінальний датасет)", "results/class_dist_original.png")
            except Exception as e:
                self._log(f"[!] Помилка при збереженні class distribution: {e}", "red")
            try:
                plot_correlation_heatmap(X, "results/correlation.png")
            except Exception as e:
                self._log(f"[!] Помилка при збереженні correlation heatmap: {e}", "red")
            self._log("\n[2/8] Розбиття 80/20 (train/val) та нормалізація...","yellow")
            val_size=config["data"]["val_size"]
            X_train,X_val,y_train,y_val=split_train_val(
                X, y, val_size=val_size, random_state=config["data"].get("random_state", 42)
            )
            self._log(f"  Train: {len(y_train)} ({100*(1-val_size):.0f}%)  Val: {len(y_val)} ({100*val_size:.0f}%)","green")
            
            # Генеруємо графіки ДО та ПІСЛЯ SMOTE
            self._log("  Генеруємо графіки розподілу класів ДО та ПІСЛЯ SMOTE...","yellow")
            y_train_before_smote = y_train.copy()
            smote_stats = {"applied": False, "before": {}, "after": {}}
            from preprocessing.augmentor import apply_smote
            from collections import Counter
            
            # Статистика ДО SMOTE
            train_counts_before = Counter(y_train_before_smote)
            smote_stats["before"] = {int(k): int(v) for k, v in train_counts_before.items()}
            
            try:
                X_train_smote, y_train_smote = apply_smote(X_train, y_train_before_smote, random_state=config["data"].get("random_state", 42))
                train_counts_after = Counter(y_train_smote)
                smote_stats["after"] = {int(k): int(v) for k, v in train_counts_after.items()}
                smote_stats["applied"] = True
                plot_smote_comparison(y_train_before_smote, y_train_smote, "results/smote_comparison.png")
                self._log(f"  ✓ Графік SMOTE (до/після): results/smote_comparison.png","green")
            except Exception as e:
                self._log(f"  [!] Помилка при генеруванні SMOTE графіку: {e}","red")
            
            self._log("  Scaler fit — лише на train; class weights — у trainer.py","green")
            X_train_sc,scaler=fit_transform(X_train)
            X_val_sc=transform(X_val,scaler)
            save_scaler(scaler,"saved_models/scaler.pkl")
            self._log("\n[3/8] Навчання MLP...","yellow")
            mlp=build_mlp(input_dim=X_train_sc.shape[1],hidden_layers=[128,64,32],dropout=0.3,
                          learning_rate=config["training"]["learning_rate"])
                          
            import tensorflow as tf
            class UICallback(tf.keras.callbacks.Callback):
                def __init__(s,app): super().__init__(); s.app=app
                def on_epoch_end(s,epoch,logs=None):
                    logs=logs or {}
                    s.app._log(f"  Epoch {epoch+1:>3} | loss={logs.get('loss',0):.4f} | acc={logs.get('accuracy',0):.4f} | val_loss={logs.get('val_loss',0):.4f} | val_acc={logs.get('val_accuracy',0):.4f}")
                    s.app.lbl_epoch.configure(text=f"Epoch {epoch+1} | val_acc={logs.get('val_accuracy',0):.4f}")
            
            # Застосовуємо SMOTE до масштабованих даних
            X_train_for_model = X_train_sc
            y_train_for_model = y_train
            if config["preprocessing"]["apply_smote"]:
                self._log("  Застосовуємо SMOTE до train даних...", "yellow")
                X_train_for_model, y_train_for_model = apply_smote(X_train_sc, y_train, random_state=config["data"].get("random_state", 42))
                self._log(f"  ✓ SMOTE завершена: train розміри {X_train_for_model.shape}", "green")
                    
            history=train_model(mlp,X_train_for_model,y_train_for_model,X_val_sc,y_val,config,"saved_models/mlp_best.keras",
                                custom_callbacks=[UICallback(self)])
            plot_training_history(history,"results/training_history_mlp.png")
            self._log("\n[4/8] Оцінювання на валідаційній вибірці...","yellow")
            y_proba=mlp.predict(X_val_sc,verbose=0); y_pred=np.argmax(y_proba,axis=1)
            from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
            acc=accuracy_score(y_val,y_pred); f1=f1_score(y_val,y_pred,average="weighted")
            try: auc=roc_auc_score(y_val,y_proba,multi_class="ovr",average="weighted")
            except: auc=0.0
            self._log(f"\n  Val Accuracy: {acc:.4f}","green")
            self._log(f"  Val F1:       {f1:.4f}","green")
            self._log(f"  Val ROC-AUC:  {auc:.4f}","green")
            plot_confusion_matrix(y_val,y_pred,"results/confusion_matrix.png")
            roc_res=plot_roc_curves(y_val,y_proba,"results/roc_curves.png")
            self._log(f"  AUC micro={roc_res['auc_micro']:.4f}  macro={roc_res['auc_macro']:.4f}","green")
            err=analyze_errors(X_val_sc,y_val,y_pred,y_proba,save_dir="results")
            self._log(f"  Помилок: {err['n_errors']} ({err['error_rate']*100:.1f}%)","green")
            self._log("\n[5/8] Порівняння MLP vs CNN...","yellow")
            _mlp2=build_mlp(input_dim=X_train_sc.shape[1],hidden_layers=[128,64,32],dropout=0.3,
                            learning_rate=config["training"]["learning_rate"])
            _cnn=build_cnn(input_dim=X_train_sc.shape[1],learning_rate=config["training"]["learning_rate"])
            arch=compare_architectures({"MLP":_mlp2,"CNN":_cnn},X_train_sc,y_train,X_val_sc,y_val,
                                       X_val_sc,y_val,config,save_dir="results")
            for aname,ares in arch.items():
                self._log(f"  {aname}: acc={ares['accuracy']:.4f}  f1={ares['f1_weighted']:.4f}","green")
            
            self._log("\n[6/8] Ablation експерименти...","yellow")
            from tensorflow.keras import layers, models, regularizers
            ablation_results = []
            ablation_configs = [
                ("mlp_basic", {"use_class_weight": False, "l2_reg": 0.0, "dropout": 0.0}),
                ("mlp_class_weights", {"use_class_weight": True, "l2_reg": 0.0, "dropout": 0.0}),
                ("mlp_full", {"use_class_weight": True, "l2_reg": 1e-4, "dropout": config["model"].get("dropout", 0.3)}),
            ]
            
            for ablation_name, ablation_cfg in ablation_configs:
                try:
                    self._log(f"  • Експеримент: {ablation_name}...", "yellow")
                    # Будуємо модель з параметрами ablation
                    l2_reg = ablation_cfg.get("l2_reg", 1e-4)
                    dropout = ablation_cfg.get("dropout", 0.3)
                    
                    abl_model = models.Sequential([
                        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None, input_shape=(X_train_sc.shape[1],)),
                        layers.BatchNormalization(),
                        layers.Dropout(dropout) if dropout > 0 else layers.Lambda(lambda x: x),
                        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None),
                        layers.BatchNormalization(),
                        layers.Dropout(dropout) if dropout > 0 else layers.Lambda(lambda x: x),
                        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None),
                        layers.BatchNormalization(),
                        layers.Dropout(dropout) if dropout > 0 else layers.Lambda(lambda x: x),
                        layers.Dense(len(np.unique(y)), activation="softmax")
                    ])
                    abl_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
                                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                    
                    # Обчислюємо class_weight якщо потрібно
                    abl_class_weight = None
                    if ablation_cfg.get("use_class_weight", False):
                        classes = np.unique(y_train)
                        total = len(y_train)
                        abl_class_weight = {int(c): float(total / (len(classes) * np.sum(y_train == c))) for c in classes}
                    
                    # Тренуємо модель
                    abl_callbacks = [
                        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["training"]["patience"], restore_best_weights=True, verbose=0),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=config["training"].get("reduce_patience", 7), verbose=0),
                    ]
                    abl_model.fit(X_train_sc, y_train, validation_data=(X_val_sc, y_val),
                                  epochs=config["training"]["epochs"], batch_size=config["training"]["batch_size"],
                                  callbacks=abl_callbacks, class_weight=abl_class_weight, verbose=0)
                    
                    # Оцінюємо
                    abl_y_proba = abl_model.predict(X_val_sc, verbose=0)
                    abl_y_pred = np.argmax(abl_y_proba, axis=1)
                    abl_metrics = evaluate(y_val, abl_y_pred, abl_y_proba)
                    
                    ablation_results.append({
                        "name": ablation_name,
                        "accuracy": float(abl_metrics["accuracy"]),
                        "f1_weighted": float(abl_metrics["f1_weighted"]),
                        "f1_macro": float(abl_metrics["f1_macro"]),
                        "roc_auc": float(abl_metrics.get("roc_auc", 0.0)),
                        "config": ablation_cfg
                    })
                    self._log(f"    ✓ {ablation_name}: acc={abl_metrics['accuracy']:.4f} f1={abl_metrics['f1_weighted']:.4f}", "green")
                except Exception as e:
                    self._log(f"    [!] Помилка в {ablation_name}: {e}", "red")
            
            self._log("\n[7/8] Baseline порівняння...", "yellow")
            baseline=train_baselines(X_train_sc,y_train,X_val_sc,y_val,"saved_models")
            for name,res in baseline.items():
                self._log(f"  {name:<22} acc={res['accuracy']:.4f}  f1={res['f1_weighted']:.4f}")
            self._log("\n[7/8] 5-кратна крос-валідація...","yellow")
            X_all,y_all,_=load_data(csv_src)
            from preprocessing.scaler import transform as _transform
            X_all_sc=_transform(X_all,scaler)
            cv=cross_validate_mlp(X_all_sc,y_all,config,n_splits=5)
            self._log(f"  CV Accuracy: {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}","green")
            self._log(f"  CV F1:       {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}","green")
            
            self._log("\n[8/8] Генерування звіту...","yellow")
            metrics_full=evaluate(y_val,y_pred,y_proba)
            save_report(metrics_full,cv,baseline,"results/report.json",
                extra={"roc_curves":roc_res,"error_analysis":err,
                       "arch_comparison":{n:{k:v for k,v in r.items() if k not in ("history","y_pred","y_proba")}
                                          for n,r in arch.items()},
                       "ablation": {"experiments": ablation_results} if ablation_results else {},
                       "split":{"train":len(y_train),"val":len(y_val),"val_size":val_size,"dataset":csv_src},
                       "smote": smote_stats,
                       "visualizations": {
                           "class_distribution_original": "results/class_dist_original.png",
                           "smote_comparison": "results/smote_comparison.png",
                           "correlation_heatmap": "results/correlation.png",
                           "confusion_matrix": "results/confusion_matrix.png",
                           "roc_curves": "results/roc_curves.png",
                           "training_history": "results/training_history_mlp.png",
                           "error_analysis": "results/error_analysis.png"
                       }})
            self._log("  Звіт збережено: results/report.json","green")
            self._log("\n"+"="*50,"bold"); self._log("  Навчання завершено успішно!","green"); self._log("="*50,"bold")
            if ablation_results:
                self._log(f"\n  📊 Ablation результати ({len(ablation_results)} експериментів):", "green")
                for ablation in ablation_results:
                    self._log(f"    • {ablation['name']}: accuracy={ablation['accuracy']:.4f}, f1={ablation['f1_weighted']:.4f}", "green")
            self.scaler=scaler; self.model=mlp; self._model_loaded=True
            self._load_dataset_context()
            self.lbl_model_status.configure(text="● модель завантажена",fg="#CCFFCC")
            messagebox.showinfo("Готово",f"Навчання завершено!\n\nVal Accuracy: {acc:.4f}\nVal F1: {f1:.4f}\nVal ROC-AUC: {auc:.4f}\nCV Accuracy: {cv['accuracy_mean']:.4f}±{cv['accuracy_std']:.4f}\n\nРезультати в папці results/")
        except Exception as e:
            import traceback; self._log(f"\n[ПОМИЛКА] {e}","red"); self._log(traceback.format_exc(),"red")
            messagebox.showerror("Помилка навчання",str(e))
        finally:
            self.after(0,lambda:self.btn_train.configure(state="normal",text="  Почати навчання  "))
            self.after(0,self.train_progress.stop)

    def _build_batch_tab(self):
        b=self.tab_batch
        top=tk.Frame(b,bg=BG); top.pack(fill="x",padx=12,pady=10)
        card=self._card(top,"Завантажити CSV для масового аналізу")
        row=tk.Frame(card,bg=CARD); row.pack(fill="x")
        tk.Label(row,text="CSV файл:",bg=CARD,fg=TEXT,font=("Segoe UI",9)).pack(side="left")
        self.var_batch_csv=tk.StringVar(value=DATA_PATH)
        tk.Entry(row,textvariable=self.var_batch_csv,width=52,font=("Segoe UI",9),relief="flat",
                 bg="#F0EFF8",highlightthickness=1,highlightbackground=BORDER).pack(side="left",padx=6)
        tk.Button(row,text="Огляд…",bg=BG,fg=ACCENT,font=("Segoe UI",9),relief="flat",cursor="hand2",
                  command=self._browse_batch).pack(side="left")
        tk.Button(top,text="  Аналізувати файл  ",bg=ACCENT,fg="white",
                  font=("Segoe UI",11,"bold"),relief="flat",cursor="hand2",activebackground=ACCENT2,
                  padx=10,pady=7,command=self._run_batch).pack(anchor="w",padx=8,pady=(10,0))
        self.batch_stats=tk.Frame(b,bg=BG); self.batch_stats.pack(fill="x",padx=12,pady=6)
        tbl_frame=self._card(b,"Результати"); tbl_frame.pack(fill="both",expand=True,padx=12,pady=(0,6))
        cols=("#","Флуор.","Колор.","SPR","Темп.","Хлор.","Волог.","Погл.","VOC","Стан","Впевн.")
        self.tree=ttk.Treeview(tbl_frame,columns=cols,show="headings",height=16)
        ttk.Style().configure("Treeview",font=("Segoe UI",9),rowheight=22)
        ttk.Style().configure("Treeview.Heading",font=("Segoe UI",9,"bold"))
        for col in cols:
            w=40 if col=="#" else 120 if col=="Стан" else 80 if col=="Впевн." else 55
            self.tree.heading(col,text=col); self.tree.column(col,width=w,anchor="center")
        for i in range(4): self.tree.tag_configure(f"class{i}",background=CLASS_BG[i])
        sb=ttk.Scrollbar(tbl_frame,orient="vertical",command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set); sb.pack(side="right",fill="y"); self.tree.pack(fill="both",expand=True)
        tk.Button(b,text="Зберегти результати як CSV",bg=BG,fg=ACCENT,font=("Segoe UI",9),
                  relief="flat",cursor="hand2",command=self._export_results).pack(anchor="e",padx=12,pady=(0,8))
        self._batch_results=[]

    def _browse_batch(self):
        path=filedialog.askopenfilename(filetypes=[("CSV файли","*.csv"),("Всі файли","*.*")])
        if path: self.var_batch_csv.set(path)

    def _run_batch(self):
        if not self._model_loaded:
            messagebox.showwarning("Модель не завантажена","Спочатку навчіть модель."); return
        import pandas as pd
        path=self.var_batch_csv.get().strip()
        if not os.path.exists(path): messagebox.showerror("Помилка",f"Файл не знайдено:\n{path}"); return
        df=pd.read_csv(path)
        missing=[c for c in FEATURE_COLS if c not in df.columns]
        if missing: messagebox.showerror("Помилка",f"Відсутні колонки:\n{', '.join(missing)}"); return
        X=df[FEATURE_COLS].values.astype(np.float32)
        X_sc=self.scaler.transform(X); proba=self.model.predict(X_sc,verbose=0); preds=np.argmax(proba,axis=1)
        for row in self.tree.get_children(): self.tree.delete(row)
        self._batch_results=[]; counts=[0,0,0,0]
        for i,(vals,pred,prob) in enumerate(zip(X,preds,proba)):
            conf=prob[pred]*100; tag=f"class{pred}"
            row_data=(i+1,)+tuple(f"{v:.2f}" for v in vals)+(CLASS_NAMES[pred],f"{conf:.1f}%")
            self.tree.insert("","end",values=row_data,tags=(tag,)); counts[pred]+=1
            self._batch_results.append({**{FEATURE_COLS[j]:vals[j] for j in range(8)},
                                         "predicted_class":pred,"predicted_name":CLASS_NAMES[pred],"confidence":conf})
        for w in self.batch_stats.winfo_children(): w.destroy()
        total=len(X)
        tk.Label(self.batch_stats,text=f"Всього: {total}",bg=BG,fg=TEXT,font=("Segoe UI",10,"bold")).pack(side="left",padx=(0,20))
        for i,(name,cnt) in enumerate(zip(CLASS_NAMES,counts)):
            tk.Label(self.batch_stats,text=f"{name}: {cnt} ({cnt/total*100:.0f}%)",
                     bg=CLASS_BG[i],fg=CLASS_COLORS[i],font=("Segoe UI",9,"bold"),padx=8,pady=3).pack(side="left",padx=4)
        if "plant_health_status" in df.columns:
            from sklearn.metrics import accuracy_score
            acc=accuracy_score(df["plant_health_status"].astype(int).values,preds)
            tk.Label(self.batch_stats,text=f"Accuracy: {acc:.4f}",bg=BG,fg="#1D9E75",font=("Segoe UI",10,"bold")).pack(side="left",padx=16)

    def _export_results(self):
        if not self._batch_results: messagebox.showinfo("Немає даних","Спочатку виконайте аналіз."); return
        import pandas as pd
        path=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV файли","*.csv")],initialfile="results_batch.csv")
        if path:
            pd.DataFrame(self._batch_results).to_csv(path,index=False,encoding="utf-8-sig")
            messagebox.showinfo("Збережено",f"Результати збережено:\n{path}")

    def _try_load_model_silent(self):
        def _load():
            try:
                if os.path.exists("saved_models/mlp_best.keras") and os.path.exists("saved_models/scaler.pkl"):
                    from preprocessing.scaler import load_scaler
                    from models.predictor import load_model
                    self.scaler=load_scaler("saved_models/scaler.pkl")
                    self.model=load_model("saved_models/mlp_best.keras")
                    self._model_loaded=True
                    self.after(0,lambda:self.lbl_model_status.configure(text="● модель завантажена",fg="#CCFFCC"))
            except Exception: pass
        threading.Thread(target=_load,daemon=True).start()

    def _card(self, parent, title=None):
        outer=tk.Frame(parent,bg=BG,pady=4); outer.pack(fill="x",pady=2)
        if title: tk.Label(outer,text=title,bg=BG,fg=MUTED,font=("Segoe UI",8,"bold")).pack(anchor="w",padx=2,pady=(0,2))
        inner=tk.Frame(outer,bg=CARD,relief="flat",highlightthickness=1,highlightbackground=BORDER,padx=12,pady=10)
        inner.pack(fill="x"); return inner

if __name__ == "__main__":
    PlantHealthApp().mainloop()