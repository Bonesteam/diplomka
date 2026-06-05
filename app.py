import os, sys, threading, numpy as np, json, datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATA_SRC = r"data/plant_health_biosensor_15k.csv"
DATA_DST = "data/plant_health_biosensor_15k.csv"
HISTORY_FILE = "analysis_history.json"

FEATURE_COLS = ["fluorescence_intensity","colorimetric_index","spr_signal_strength",
                "leaf_temperature","chlorophyll_content","moisture_level",
                "light_absorption_ratio","volatile_organic_compounds"]
FEATURE_UA = ["Флуоресценція","Колориметричний індекс","SPR-сигнал","Температура листа (°C)",
              "Вміст хлорофілу","Рівень вологості","Поглинання світла","Леткі сполуки (VOC)"]

# Нормальні діапазони для перевірки
NORMAL_RANGES = [
    (45, 60, "норма: 45–60"),
    (0.55, 0.70, "норма: 0.55–0.70"),
    (85, 115, "норма: 85–115"),
    (20, 30, "норма: 20–30"),
    (33, 50, "норма: 33–50"),
    (25, 35, "норма: 25–35"),
    (0.65, 0.74, "норма: 0.65–0.74"),
    (10, 19, "норма: 10–19")
]

# Загальні діапазони для полів введення
FEATURE_RANGES = [(15,90,"норма: 45–60"),(0.25,0.95,"норма: 0.55–0.70"),
                  (40,180,"норма: 85–115"),(9,41,"норма: 20–30"),
                  (10,72,"норма: 33–50"),(13,48,"норма: 25–35"),
                  (0.50,0.87,"норма: 0.65–0.74"),(0,29,"норма: 10–19")]

CLASS_NAMES  = ["Критичний стрес","Помірний стрес","Легкий стрес","Здорова рослина"]
CLASS_COLORS = ["#E24B4A","#EF9F27","#639922","#1D9E75"]
CLASS_BG     = ["#FCEBEB","#FAEEDA","#EAF3DE","#E1F5EE"]

PRESETS = {
    # critical: many values intentionally outside NORMAL_RANGES
    "Критичний стрес": [30.0, 0.30, 40.0, 35.0, 20.0, 15.0, 0.50, 5.0],
    "Помірний стрес":  [50.2, 0.61, 124.93, 24.68, 34.15, 30.10, 0.70, 15.43],
    "Легкий стрес":    [50.9, 0.46, 94.73, 24.12, 41.04, 25.13, 0.70, 15.34],
    # healthy: use dataset mean for healthy class to avoid false deviations
    "Здорова рослина": [48.91, 0.61, 92.46, 24.82, 39.62, 29.96, 0.70, 14.78],
}

ACCENT="#534AB7"; ACCENT2="#7F77DD"; BG="#F8F8F8"; CARD="#FFFFFF"; BORDER="#E0DED8"; TEXT="#1A1A1A"; MUTED="#6B6B68"

from analysis_logic import classify, calibrated_confidence


class PlantHealthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Оцінювання стану здоров'я рослин")
        self.geometry("1080x720"); self.minsize(900,620); self.configure(bg=BG)
        self.model=None; self.scaler=None; self._model_loaded=False
        self.history = self._load_history()
        self.class_means = None
        self.class_stds = None
        self._compute_class_stats()
        self._build_ui(); self._try_load_model_silent()

    def _compute_class_stats(self):
        try:
            from preprocessing.loader import FEATURE_COLS, TARGET_COL
            import pandas as pd
            path = DATA_DST if os.path.exists(DATA_DST) else (DATA_SRC if os.path.exists(DATA_SRC) else None)
            if path is None:
                return
            df = pd.read_csv(path)
            grp = df.groupby(TARGET_COL)[FEATURE_COLS]
            means = grp.mean()
            stds = grp.std().replace(0, 1e-6)
            # align by class index 0..n-1
            self.class_means = means.values
            self.class_stds = stds.values
        except Exception:
            self.class_means = None
            self.class_stds = None

    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except: return []
        return []

    def _save_to_history(self, values, result_class, confidence, probs):
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "values": values,
            "result_class": result_class,
            "confidence": confidence,
            "probabilities": [float(p) for p in probs]
        }
        self.history.insert(0, entry)
        if len(self.history) > 100:
            self.history = self.history[:100]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        if hasattr(self, 'history_tree'):
            self._update_history_table()

    def _update_history_table(self):
        for row in self.history_tree.get_children():
            self.history_tree.delete(row)
        for entry in self.history[:50]:
            vals = entry["values"]
            self.history_tree.insert("", "end", values=(
                entry["timestamp"],
                f"{vals[0]:.1f}", f"{vals[1]:.2f}", f"{vals[2]:.1f}",
                f"{vals[3]:.1f}", f"{vals[4]:.1f}", f"{vals[5]:.1f}",
                f"{vals[6]:.2f}", f"{vals[7]:.1f}",
                entry["result_class"], f"{entry['confidence']:.1f}%"
            ))

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
        self.tab_history  = tk.Frame(nb, bg=BG)
        self.tab_info    = tk.Frame(nb, bg=BG)
        
        nb.add(self.tab_predict, text="  Аналіз рослини  ")
        nb.add(self.tab_train,   text="  Навчання моделі  ")
        nb.add(self.tab_batch,   text="  Пакетний аналіз CSV  ")
        nb.add(self.tab_history, text="  Історія аналізів  ")
        nb.add(self.tab_info,    text="  Довідка  ")
        
        self._build_predict_tab()
        self._build_train_tab()
        self._build_batch_tab()
        self._build_history_tab()
        self._build_info_tab()

    # ==================== СТОРІНКА ІСТОРІЯ АНАЛІЗІВ ====================
    def _build_history_tab(self):
        t = self.tab_history
        top = tk.Frame(t, bg=BG); top.pack(fill="x", padx=12, pady=10)
        
        tk.Label(top, text="📋 Історія аналізів", bg=BG, fg=ACCENT, 
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        
        btn_frame = tk.Frame(top, bg=BG); btn_frame.pack(side="right")
        tk.Button(btn_frame, text="🗑 Очистити історію", bg=BG, fg="#E24B4A",
                  font=("Segoe UI", 9), relief="flat", cursor="hand2",
                  command=self._clear_history).pack(side="left", padx=5)
        tk.Button(btn_frame, text="📎 Експортувати CSV", bg=BG, fg=ACCENT,
                  font=("Segoe UI", 9), relief="flat", cursor="hand2",
                  command=self._export_history).pack(side="left")
        
        # Таблиця історії
        cols = ("Дата/Час", "Флуор.", "Колор.", "SPR", "Темп.", "Хлор.", "Волог.", "Погл.", "VOC", "Результат", "Впевн.")
        self.history_tree = ttk.Treeview(t, columns=cols, show="headings", height=18)
        
        for col in cols:
            w = 140 if col == "Дата/Час" else 70 if col == "Впевн." else 60
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=w, anchor="center")
        
        for i in range(4):
            self.history_tree.tag_configure(f"class{i}", background=CLASS_BG[i])
        
        sb = ttk.Scrollbar(t, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=sb.set)
        
        self.history_tree.pack(side="left", fill="both", expand=True, padx=12, pady=(0,10))
        sb.pack(side="right", fill="y", pady=(0,10))
        
        self._update_history_table()
        
        # Кнопка деталей
        btn_details = tk.Button(t, text="🔍 Показати деталі вибраного аналізу", 
                                 bg=ACCENT, fg="white", font=("Segoe UI", 10),
                                 relief="flat", cursor="hand2", command=self._show_history_details)
        btn_details.pack(pady=(0, 10))

    def _clear_history(self):
        if messagebox.askyesno("Підтвердження", "Очистити всю історію аналізів?"):
            self.history = []
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
            self._update_history_table()
            messagebox.showinfo("Готово", "Історію очищено")

    def _export_history(self):
        if not self.history:
            messagebox.showinfo("Немає даних", "Історія порожня")
            return
        import pandas as pd
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV файли","*.csv")])
        if path:
            df = pd.DataFrame(self.history)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("Готово", f"Збережено: {path}")

    def _show_history_details(self):
        selected = self.history_tree.selection()
        if not selected:
            messagebox.showwarning("Виберіть запис", "Оберіть аналіз з таблиці")
            return
        idx = self.history_tree.index(selected[0])
        if idx < len(self.history):
            entry = self.history[idx]
            details = f"📅 Дата: {entry['timestamp']}\n\n"
            details += "📊 Показники:\n"
            for i, ua in enumerate(FEATURE_UA):
                details += f"  {ua}: {entry['values'][i]:.3f}\n"
            details += f"\n🏷 Результат: {entry['result_class']}\n"
            details += f"📈 Впевненість: {entry['confidence']:.1f}%\n"
            messagebox.showinfo("Деталі аналізу", details)

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
            ("Флуоресценція", "45–60", "відносні од."),
            ("Колориметричний індекс", "0.55–0.70", "одиниці"),
            ("SPR-сигнал", "85–115", "нм"),
            ("Температура листа", "20–30", "°C"),
            ("Вміст хлорофілу", "33–50", "мг/г"),
            ("Рівень вологості", "25–35", "%"),
            ("Поглинання світла", "0.65–0.74", "коеф."),
            ("Леткі сполуки (VOC)", "10–19", "ppm")
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
        for name in PRESETS:
            tk.Button(card0, text=name, bg=BG, fg=ACCENT, font=("Segoe UI",9), relief="flat",
                      bd=0, cursor="hand2", activeforeground=ACCENT2,
                      command=lambda n=name: self._load_preset(n)).pack(anchor="w", pady=1)
        
        card1 = self._card(left, "Дані біосенсорів"); self.entries = []
        for ua, (lo, hi, hint) in zip(FEATURE_UA, FEATURE_RANGES):
            row = tk.Frame(card1, bg=CARD); row.pack(fill="x", pady=3)
            tk.Label(row, text=ua, bg=CARD, fg=TEXT, font=("Segoe UI",9,"bold"), anchor="w", width=28).pack(side="left")
            var = tk.StringVar(value=str(round((lo+hi)/2, 2)))
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
        res = classify(values, probs, NORMAL_RANGES)
        final_cls = res["final_cls"]
        display_probs = res["final_probs"]
        conf = res["confidence"]
        statuses = res["statuses"]
        model_cls = res["model_cls"]
        model_conf = calibrated_confidence(res["model_probs_soft"])
        n_clear = res["n_clear_deviations"]
        blend_w = res["blend_w"]

        color = CLASS_COLORS[final_cls]
        bg_c = CLASS_BG[final_cls]
        banner = tk.Frame(self.result_frame, bg=bg_c, bd=0); banner.pack(fill="x", pady=(0,10))
        tk.Label(banner, text=CLASS_NAMES[final_cls], bg=bg_c, fg=color, font=("Segoe UI",18,"bold")).pack(pady=(14,2))
        conf_txt = f"Впевненість: {conf:.1f}%"
        margin = float(display_probs[final_cls] - np.partition(display_probs, -2)[-2])
        if blend_w < 0.9 and margin < 0.18:
            conf_txt += "  (близькі класи — змінюйте показники поступово)"
        elif blend_w >= 0.9 and n_clear >= 5:
            conf_txt += "  (оцінка за показниками — значення далеко від норми)"
        tk.Label(banner, text=conf_txt, bg=bg_c, fg=color, font=("Segoe UI",11)).pack(pady=(0,14))

        card_prob = self._card(self.result_frame, "Ймовірності класів")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, display_probs)):
            pct = prob * 100; row = tk.Frame(card_prob, bg=CARD); row.pack(fill="x", pady=3)
            tk.Label(row, text=name, bg=CARD, fg=TEXT, font=("Segoe UI",9), width=18, anchor="w").pack(side="left")
            track = tk.Frame(row, bg=BORDER, height=12, width=220); track.pack(side="left", padx=6); track.pack_propagate(False)
            fill_w = int(pct/100*220)
            if fill_w > 0: tk.Frame(track, bg=CLASS_COLORS[i], height=12, width=fill_w).place(x=0, y=0)
            tk.Label(row, text=f"{pct:.1f}%", bg=CARD, fg=MUTED, font=("Segoe UI",9), width=6).pack(side="left")
            if i == final_cls: tk.Label(row, text="◄", bg=CARD, fg=color, font=("Segoe UI",9)).pack(side="left")

        card_sens = self._card(self.result_frame, "Аналіз показників")
        status_ui = {
            "ok": ("#1D9E75", "✔ норма"),
            "borderline": ("#E6A800", "⚠ на межі"),
            "deviation": ("#E24B4A", "✘ відхилення"),
        }

        for i, (ua, (_, _, hint), val) in enumerate(zip(FEATURE_UA, NORMAL_RANGES, values)):
            st, _ = statuses[i]
            val_color, dt = status_ui[st]
            row = tk.Frame(card_sens, bg=CARD); row.pack(fill="x", pady=2)
            tk.Label(row, text=ua, bg=CARD, fg=TEXT, font=("Segoe UI",9), width=32, anchor="w").pack(side="left")
            tk.Label(row, text=f"{val:.3f}", bg=CARD, fg=val_color, font=("Segoe UI",9,"bold"), width=8).pack(side="left")
            tk.Label(row, text=dt, bg=CARD, fg=val_color, font=("Segoe UI",9), width=14).pack(side="left")
            tk.Label(row, text=hint, bg=CARD, fg=MUTED, font=("Segoe UI",8)).pack(side="left", padx=4)

        summary_lines = []
        for ua, (st, _) in zip(FEATURE_UA, statuses):
            if st == "deviation":
                summary_lines.append(f"✘ {ua}")
            elif st == "borderline":
                summary_lines.append(f"⚠ {ua}")
        if not summary_lines:
            summary_lines.append("Усі показники в нормі.")

        model_line = f"Модель (пом'якшено): {CLASS_NAMES[model_cls]} ({model_conf:.1f}%)"
        if blend_w > 0.05:
            rule_line = (
                f"Показники: явних відхилень {n_clear}/8, "
                f"внесок правил {blend_w*100:.0f}%"
            )
        else:
            rule_line = "Показники: усі в нормі — рішення за моделлю"

        summary_text = "\n".join([model_line, rule_line, "", *summary_lines])
        summ = tk.Label(self.result_frame, text=summary_text,
                       bg=BG, fg=TEXT, font=("Segoe UI",10), justify="left")
        summ.pack(padx=8, pady=(6,12), anchor="w")

        self._save_to_history(values, CLASS_NAMES[final_cls], conf, display_probs.tolist())

    def _load_preset(self, name):
        for var,v in zip(self.entries,PRESETS[name]): var.set(str(v))

    def _run_predict(self):
        if not self._model_loaded:
            messagebox.showwarning("Модель не завантажена","Спочатку навчіть модель.")
            return
        try:
            values = [float(v.get()) for v in self.entries]
        except ValueError:
            messagebox.showerror("Помилка","Перевірте значення — мають бути числами.")
            return
        # Deterministic prediction: do not add noise, do not apply temperature scaling
        # or manual probability adjustments — same input should yield same output.
        X = np.array([values], dtype=np.float32)
        X_sc = self.scaler.transform(X)
        probs = self.model.predict(X_sc, verbose=0)[0]
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
            
        train_path = cfg.get("data", {}).get("path", DATA_SRC)
        test_path = cfg.get("data", {}).get("test_path", "plant_health_biosensor_dataset.csv")
        epochs = str(cfg.get("training", {}).get("epochs", 100))
        batch_size = str(cfg.get("training", {}).get("batch_size", 32))
        lr = str(cfg.get("training", {}).get("learning_rate", 0.0008))
        patience = str(cfg.get("training", {}).get("patience", 15))
        val_size = str(cfg.get("data", {}).get("val_size", 0.20))

        card_data=self._card(top,"Датасет")
        
        # Row 1: Train/Val CSV
        row1=tk.Frame(card_data,bg=CARD); row1.pack(fill="x", pady=2)
        tk.Label(row1,text="Train/Val CSV:",bg=CARD,fg=TEXT,font=("Segoe UI",9), width=12, anchor="w").pack(side="left")
        self.var_csv=tk.StringVar(value=train_path)
        tk.Entry(row1,textvariable=self.var_csv,width=52,font=("Segoe UI",9),relief="flat",
                 bg="#F0EFF8",highlightthickness=1,highlightbackground=BORDER).pack(side="left",padx=6)
        tk.Button(row1,text="Огляд…",bg=BG,fg=ACCENT,font=("Segoe UI",9),relief="flat",cursor="hand2",
                  command=self._browse_csv).pack(side="left")
                  
        # Row 2: Test CSV
        row2=tk.Frame(card_data,bg=CARD); row2.pack(fill="x", pady=2)
        tk.Label(row2,text="Test CSV:",bg=CARD,fg=TEXT,font=("Segoe UI",9), width=12, anchor="w").pack(side="left")
        self.var_test_csv=tk.StringVar(value=test_path)
        tk.Entry(row2,textvariable=self.var_test_csv,width=52,font=("Segoe UI",9),relief="flat",
                 bg="#F0EFF8",highlightthickness=1,highlightbackground=BORDER).pack(side="left",padx=6)
        tk.Button(row2,text="Огляд…",bg=BG,fg=ACCENT,font=("Segoe UI",9),relief="flat",cursor="hand2",
                  command=self._browse_test_csv).pack(side="left")

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
            
        self.var_smote=tk.BooleanVar(value=True)
        tk.Checkbutton(card_hp,text="Застосувати SMOTE (балансування класів)",variable=self.var_smote,
                       bg=CARD,fg=TEXT,font=("Segoe UI",9),activebackground=CARD).pack(anchor="w",pady=(8,0))
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

    def _browse_test_csv(self):
        path=filedialog.askopenfilename(filetypes=[("CSV файли","*.csv"),("Всі файли","*.*")])
        if path: self.var_test_csv.set(path)

    def _log(self, msg, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", msg+"\n", tag if tag else "")
        self.log.see("end"); self.log.configure(state="disabled")

    def _run_training_thread(self):
        self.btn_train.configure(state="disabled",text="Навчання…"); self.train_progress.start(12)
        self.log.configure(state="normal"); self.log.delete("1.0","end"); self.log.configure(state="disabled")
        threading.Thread(target=self._run_training,daemon=True).start()

    def _run_training(self):
        import shutil, yaml
        try:
            self._log("="*50,"bold"); self._log("  Запуск навчання нейромережі","bold"); self._log("="*50,"bold")
            csv_src=self.var_csv.get().strip()
            if not os.path.exists(csv_src): self._log(f"[!] Файл не знайдено: {csv_src}","red"); return
            os.makedirs("data",exist_ok=True)
            try:
                if os.path.abspath(csv_src) != os.path.abspath(DATA_DST):
                    shutil.copy(csv_src, DATA_DST)
            except shutil.SameFileError:
                pass  # файл вже на місці, копіювання не потрібне
            self._log(f"Датасет: {csv_src}","green")
            from preprocessing.loader import load_data
            from preprocessing.splitter import split_data
            from preprocessing.scaler import fit_transform,transform,save_scaler
            from preprocessing.augmentor import apply_smote
            from models.mlp_model import build_mlp
            from models.cnn_model import build_cnn
            from models.trainer import train_model
            from models.predictor import predict
            from models.baseline_ml import train_baselines
            from evaluation.metrics import evaluate
            from evaluation.confusion import plot_confusion_matrix
            from evaluation.reporter import save_report,print_comparison
            from evaluation.error_analysis import analyze_errors
            from evaluation.arch_comparison import compare_architectures
            from evaluation.cross_val import cross_validate_mlp
            from visualization.plots import plot_training_history,plot_class_distribution,plot_comparison_bar
            from visualization.feature_imp import plot_correlation_heatmap
            from visualization.roc_curves import plot_roc_curves
            from preprocessing.loader import load_data as _load_data_for_cv
            with open("config.yaml") as f: config=yaml.safe_load(f)
            config["training"]["epochs"]=int(self.hp_vars[0].get())
            config["training"]["batch_size"]=int(self.hp_vars[1].get())
            config["training"]["learning_rate"]=float(self.hp_vars[2].get())
            config["training"]["patience"]=int(self.hp_vars[3].get())
            config["data"]["val_size"]=float(self.hp_vars[4].get())
            config["data"]["path"]=csv_src
            
            test_csv=self.var_test_csv.get().strip()
            config["data"]["test_path"]=test_csv
            config["preprocessing"]["apply_smote"]=self.var_smote.get()
            
            # Save updated config
            with open("config.yaml", "w") as f:
                yaml.safe_dump(config, f)
                
            os.makedirs("results",exist_ok=True); os.makedirs("saved_models",exist_ok=True)
            self._log("\n[1/7] Завантаження даних...","yellow")
            
            # Load main Train/Val dataset
            X,y,_=load_data(DATA_DST)
            self._log(f"  Зразків Train/Val: {X.shape[0]}, ознак: {X.shape[1]}")
            
            # Load Test dataset
            has_test_file = False
            if test_csv and os.path.exists(test_csv):
                try:
                    X_test_raw, y_test, _ = load_data(test_csv)
                    self._log(f"  Зразків Test (з файлу {test_csv}): {X_test_raw.shape[0]}")
                    has_test_file = True
                except Exception as e:
                    self._log(f"[!] Помилка завантаження тестового файлу: {e}", "red")
            
            os.makedirs("results", exist_ok=True)
            try:
                plot_class_distribution(y, "Розподіл класів (вихідний)", "results/class_dist_original.png")
            except Exception as e:
                self._log(f"[!] Помилка при збереженні class distribution: {e}", "red")
            try:
                plot_correlation_heatmap(X, "results/correlation.png")
            except Exception as e:
                self._log(f"[!] Помилка при збереженні correlation heatmap: {e}", "red")
                
            self._log("\n[2/7] Розбиття та нормалізація...","yellow")
            
            from sklearn.model_selection import train_test_split
            if has_test_file:
                # Split main dataset into train/validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=config["data"]["val_size"], random_state=42, stratify=y
                )
                X_test = X_test_raw
            else:
                # Fallback to splitting main dataset into Train/Val/Test
                self._log("  [!] Тестовий файл не знайдено. Використовуємо 3-сторонній спліт.", "yellow")
                test_size_fallback = 0.15
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                    X, y, test_size=test_size_fallback, val_size=config["data"]["val_size"] * (1 - test_size_fallback), random_state=42
                )
                
            X_train_sc,scaler=fit_transform(X_train); X_val_sc=transform(X_val,scaler); X_test_sc=transform(X_test,scaler)
            save_scaler(scaler,"saved_models/scaler.pkl")
            
            if config["preprocessing"]["apply_smote"]:
                X_train_sc,y_train=apply_smote(X_train_sc,y_train)
                self._log(f"  Після SMOTE: {len(y_train)} зразків")
                try:
                    plot_class_distribution(y_train, "Після SMOTE", "results/class_dist_smote.png")
                except Exception as e:
                    self._log(f"[!] Помилка при збереженні class distribution після SMOTE: {e}", "red")
                    
            self._log(f"  Train:{len(y_train)}  Val:{len(y_val) if y_val is not None else 0}  Test:{len(y_test)}")
            
            self._log("\n[3/7] Навчання MLP...","yellow")
            mlp=build_mlp(input_dim=X_train_sc.shape[1],hidden_layers=[128,64,32],dropout=0.3,
                          learning_rate=config["training"]["learning_rate"])
                          
            import tensorflow as tf
            class UICallback(tf.keras.callbacks.Callback):
                def __init__(s,app): super().__init__(); s.app=app
                def on_epoch_end(s,epoch,logs=None):
                    logs=logs or {}
                    s.app._log(f"  Epoch {epoch+1:>3} | loss={logs.get('loss',0):.4f} | acc={logs.get('accuracy',0):.4f} | val_loss={logs.get('val_loss',0):.4f} | val_acc={logs.get('val_accuracy',0):.4f}")
                    s.app.lbl_epoch.configure(text=f"Epoch {epoch+1} | val_acc={logs.get('val_accuracy',0):.4f}")
                    
            history=train_model(mlp,X_train_sc,y_train,X_val_sc,y_val,config,"saved_models/mlp_best.keras",
                                custom_callbacks=[UICallback(self)])
            plot_training_history(history,"results/training_history_mlp.png")
            self._log("\n[4/7] Оцінювання...","yellow")
            y_proba=mlp.predict(X_test_sc,verbose=0); y_pred=np.argmax(y_proba,axis=1)
            from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
            acc=accuracy_score(y_test,y_pred); f1=f1_score(y_test,y_pred,average="weighted")
            try: auc=roc_auc_score(y_test,y_proba,multi_class="ovr",average="weighted")
            except: auc=0.0
            self._log(f"\n  Accuracy:    {acc:.4f}","green")
            self._log(f"  F1 weighted: {f1:.4f}","green")
            self._log(f"  ROC-AUC:     {auc:.4f}","green")
            plot_confusion_matrix(y_test,y_pred,"results/confusion_matrix.png")
            roc_res=plot_roc_curves(y_test,y_proba,"results/roc_curves.png")
            self._log(f"  AUC micro={roc_res['auc_micro']:.4f}  macro={roc_res['auc_macro']:.4f}","green")
            err=analyze_errors(X_test_sc,y_test,y_pred,y_proba,save_dir="results")
            self._log(f"  Помилок: {err['n_errors']} ({err['error_rate']*100:.1f}%)","green")
            self._log("\n[5/7] Порівняння MLP vs CNN...","yellow")
            _mlp2=build_mlp(input_dim=X_train_sc.shape[1],hidden_layers=[128,64,32],dropout=0.3,
                            learning_rate=config["training"]["learning_rate"])
            _cnn=build_cnn(input_dim=X_train_sc.shape[1],learning_rate=config["training"]["learning_rate"])
            arch=compare_architectures({"MLP":_mlp2,"CNN":_cnn},X_train_sc,y_train,X_val_sc,y_val,
                                       X_test_sc,y_test,config,save_dir="results")
            for aname,ares in arch.items():
                self._log(f"  {aname}: acc={ares['accuracy']:.4f}  f1={ares['f1_weighted']:.4f}","green")
            self._log("\n[6/7] Baseline порівняння...","yellow")
            baseline=train_baselines(X_train_sc,y_train,X_test_sc,y_test,"saved_models")
            for name,res in baseline.items():
                self._log(f"  {name:<22} acc={res['accuracy']:.4f}  f1={res['f1_weighted']:.4f}")
            self._log("\n[7/7] 5-кратна крос-валідація...","yellow")
            X_all,y_all,_=_load_data_for_cv(DATA_DST)
            from preprocessing.scaler import transform as _transform
            X_all_sc=_transform(X_all,scaler)
            cv=cross_validate_mlp(X_all_sc,y_all,config,n_splits=5)
            self._log(f"  CV Accuracy: {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}","green")
            self._log(f"  CV F1:       {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}","green")
            metrics_full=evaluate(y_test,y_pred,y_proba)
            save_report(metrics_full,cv,baseline,"results/report.json",
                extra={"roc_curves":roc_res,"error_analysis":err,
                       "arch_comparison":{n:{k:v for k,v in r.items() if k not in ("history","y_pred","y_proba")}
                                          for n,r in arch.items()}})
            self._log("  Звіт збережено: results/report.json","green")
            self._log("\n"+"="*50,"bold"); self._log("  Навчання завершено успішно!","green"); self._log("="*50,"bold")
            self.scaler=scaler; self.model=mlp; self._model_loaded=True
            self.lbl_model_status.configure(text="● модель завантажена",fg="#CCFFCC")
            messagebox.showinfo("Готово",f"Навчання завершено!\n\nAccuracy: {acc:.4f}\nF1: {f1:.4f}\nROC-AUC: {auc:.4f}\nCV Accuracy: {cv['accuracy_mean']:.4f}±{cv['accuracy_std']:.4f}\n\nРезультати в папці results/")
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
        self.var_batch_csv=tk.StringVar(value=DATA_SRC)
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