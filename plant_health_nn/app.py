import os, threading, numpy as np, json, datetime
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

# Резервні пресети, якщо датасет недоступний при старті
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
    compute_normal_ranges, compute_input_ranges, compute_class_profiles,
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
        self._build_ui()
        self._try_load_model_silent()

    def _dataset_path(self):
        if os.path.exists(DATA_PATH):
            return DATA_PATH
        return None

    def _load_dataset_context(self):
        """Завантаження еталонів, пресетів і діапазонів з датасету."""
        from preprocessing.loader import TARGET_COL
        import pandas as pd

        path = self._dataset_path()
        if path is None:
            return
        try:
            df = pd.read_csv(path)
            self.normal_ranges = compute_normal_ranges(df, FEATURE_COLS, TARGET_COL)
            self.feature_ranges = compute_input_ranges(df, FEATURE_COLS)
            self.class_means, self.class_stds = compute_class_profiles(df, FEATURE_COLS, TARGET_COL)
            means = df.groupby(TARGET_COL)[FEATURE_COLS].mean()
            presets = {}
            for i, name in enumerate(CLASS_NAMES):
                sub = df[df[TARGET_COL] == i][FEATURE_COLS]
                if len(sub) == 0:
                    presets[name] = FALLBACK_PRESETS.get(name, [])
                    continue
                centroid = means.loc[i]
                dists = ((sub - centroid) ** 2).sum(axis=1)
                best_idx = dists.idxmin()
                presets[name] = [round(float(v), 2) for v in df.loc[best_idx, FEATURE_COLS].tolist()]
            self.presets = presets
        except Exception:
            self.presets = dict(FALLBACK_PRESETS)

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

    def _load_report_metrics(self):
        """Метрики з останнього навчання (results/report.json)."""
        try:
            if os.path.exists("results/report.json"):
                with open("results/report.json", encoding="utf-8") as f:
                    report = json.load(f)
                nn = report.get("neural_network", {})
                return {
                    "accuracy": nn.get("accuracy"),
                    "f1": nn.get("f1_weighted"),
                    "roc_auc": nn.get("roc_auc"),
                }
        except Exception:
            pass
        return {}

    def _load_config_snapshot(self):
        import yaml
        if not os.path.exists("config.yaml"):
            return {}
        try:
            with open("config.yaml", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _refresh_info_tab(self):
        for w in self.tab_info.winfo_children():
            w.destroy()
        self._build_info_tab()

    # ==================== СТОРІНКА ДОВІДКА ====================
    def _build_info_tab(self):
        info = tk.Frame(self.tab_info, bg=BG)
        info.pack(fill="both", expand=True, padx=16, pady=12)

        cfg = self._load_config_snapshot()
        metrics = self._load_report_metrics()
        dataset_path = cfg.get("data", {}).get("path", DATA_PATH)
        val_split = cfg.get("training", {}).get("validation_split", 0.2)
        lr = cfg.get("training", {}).get("learning_rate", 0.001)
        dropout = cfg.get("model", {}).get("dropout", 0.3)

        tk.Label(info, text="📊 Про систему", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 10))

        if metrics.get("accuracy") is not None:
            desc_txt = (
                f"Система класифікує стан рослини за 8 біосенсорними показниками "
                f"(MLP + профіль ознак). Останнє навчання: test accuracy "
                f"{metrics['accuracy']*100:.1f}%, ROC-AUC {metrics.get('roc_auc', 0):.3f}."
            )
        else:
            desc_txt = (
                "Система класифікує стан рослини за 8 біосенсорними показниками. "
                "Спочатку навчіть модель на вкладці «Навчання моделі»."
            )
        tk.Label(info, text=desc_txt, bg=BG, fg=TEXT, font=("Segoe UI", 11),
                 wraplength=750, justify="left").pack(anchor="w", pady=(0, 8))
        tk.Label(info, text=f"Датасет: {dataset_path}", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 12))

        tk.Label(info, text="🔬 Біосенсорні показники", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        table_frame = tk.Frame(info, bg=CARD, relief="flat", bd=1,
                               highlightbackground=BORDER, highlightthickness=1)
        table_frame.pack(fill="x", pady=(0, 15))

        headers = ("Ознака", "Колонка CSV", "Профіль здорової", "Допустимо для вводу")
        for j, col in enumerate(headers):
            tk.Label(table_frame, text=col, bg=ACCENT, fg="white",
                     font=("Segoe UI", 10, "bold"), padx=8, pady=6).grid(row=0, column=j, sticky="ew")

        for i, (ua, col_name) in enumerate(zip(FEATURE_UA, FEATURE_COLS)):
            _, _, normal_hint = self.normal_ranges[i]
            _, _, input_hint = self.feature_ranges[i]
            bg_color = "#F8F9FA" if i % 2 == 0 else "#FFFFFF"
            row_vals = (ua, col_name, normal_hint, input_hint)
            for j, val in enumerate(row_vals):
                tk.Label(table_frame, text=val, bg=bg_color, fg=TEXT,
                         font=("Segoe UI", 9), padx=8, pady=4, anchor="w",
                         wraplength=180 if j == 0 else 220).grid(row=i + 1, column=j, sticky="ew")

        for j in range(len(headers)):
            table_frame.columnconfigure(j, weight=1)

        tk.Label(info, text="Профіль здорової — середнє ± 0.8σ для класу «Здорова рослина». "
                 "Допустимо — 1–99 перцентиль датасету (як на формі вводу).",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8), wraplength=750,
                 justify="left").pack(anchor="w", pady=(0, 12))

        tk.Label(info, text="🧠 Про нейронну мережу", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        model_frame = tk.Frame(info, bg=CARD, relief="flat", bd=1,
                               highlightbackground=BORDER, highlightthickness=1)
        model_frame.pack(fill="x", pady=(0, 15))

        acc_line = (
            f"• Test accuracy (останнє навчання): {metrics['accuracy']*100:.1f}%"
            if metrics.get("accuracy") is not None else
            "• Test accuracy: навчіть модель для оцінки"
        )
        auc_line = (
            f"• ROC-AUC (зважений): {metrics['roc_auc']:.3f}"
            if metrics.get("roc_auc") is not None else
            "• ROC-AUC: —"
        )
        model_info = [
            "• Архітектура: MLP (128 → 64 → 32 нейрони)",
            "• Активація: ReLU  |  Dropout + BatchNormalization",
            f"• Оптимізатор: Adam (learning rate = {lr})",
            f"• Dropout: {dropout}",
            "• Функція втрат: sparse_categorical_crossentropy",
            acc_line,
            auc_line,
        ]
        for line in model_info:
            tk.Label(model_frame, text=line, bg=CARD, fg=TEXT, font=("Segoe UI", 10),
                     anchor="w", padx=12, pady=3).pack(fill="x")

        tk.Label(info, text="📁 Розбиття датасету", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        split_frame = tk.Frame(info, bg=CARD, relief="flat", bd=1,
                               highlightbackground=BORDER, highlightthickness=1)
        split_frame.pack(fill="x", pady=(0, 15))

        split_lines = [
            "• Один CSV → sklearn train_test_split",
            "• Train: 80%  |  Test: 20% (ізольований, фінальна оцінка)",
            f"• Val на графіках: Keras validation_split={val_split} (автоматично з train)",
            "  ⚠ val_accuracy може бути вищою за train: Dropout вимкнений під час валідації.",
            "• Окремого validation-файлу немає",
            "• SMOTE застосовується лише до train (якщо увімкнено)",
        ]
        for line in split_lines:
            tk.Label(split_frame, text=line, bg=CARD, fg=TEXT, font=("Segoe UI", 10),
                     anchor="w", padx=12, pady=3).pack(fill="x")

        tk.Label(info, text="📖 Як користуватися", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        inst_frame = tk.Frame(info, bg=CARD, relief="flat", bd=1,
                              highlightbackground=BORDER, highlightthickness=1)
        inst_frame.pack(fill="x")

        instructions = [
            "1. Навчіть модель на вкладці «Навчання моделі» (або завантажте saved_models/)",
            "2. Введіть 8 показників на «Аналіз рослини» або оберіть шаблон класу",
            "3. Натисніть «Аналізувати» — система покаже клас і впевненість",
            "4. Для CSV з колонками як у датасеті — «Пакетний аналіз CSV»",
            "5. Графіки та звіт — у папці results/",
        ]
        for line in instructions:
            tk.Label(inst_frame, text=line, bg=CARD, fg=TEXT, font=("Segoe UI", 10),
                     anchor="w", padx=12, pady=3).pack(fill="x")

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
        
        btn_frame = tk.Frame(left, bg=BG)
        btn_frame.pack(fill="x", padx=8, pady=(10,0))

        self.btn_analyze = tk.Button(btn_frame, text="  Аналізувати  ", bg=ACCENT, fg="white",
            font=("Segoe UI",11,"bold"), relief="flat", cursor="hand2",
            activebackground=ACCENT2, padx=10, pady=8, command=self._run_predict)
        self.btn_analyze.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_sensitivity = tk.Button(btn_frame, text="📊 Аналіз чутливості", bg="#1D9E75", fg="white",
            font=("Segoe UI",11,"bold"), relief="flat", cursor="hand2",
            activebackground="#1D9E75", padx=10, pady=8, command=self._open_sensitivity_analysis)
        self.btn_sensitivity.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.result_frame = tk.Frame(right, bg=BG); self.result_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self._show_placeholder()

    def _show_placeholder(self):
        for w in self.result_frame.winfo_children(): w.destroy()
        tk.Label(self.result_frame, text="Введіть дані біосенсорів\nі натисніть «Аналізувати»",
                 bg=BG, fg=MUTED, font=("Segoe UI",13)).pack(expand=True)

    def _show_result(self, values, probs):
        for w in self.result_frame.winfo_children(): w.destroy()
        res = classify(values, probs, self.normal_ranges,
                       class_means=self.class_means, class_stds=self.class_stds)
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
        
        # Поліпшена логіка невизначеності
        if conf < 50:
            conf_txt += "  🔴 НИЗЬКА впевненість! Результат невизначений"
        elif margin < 0.10:
            conf_txt += "  ⚠ Результати дуже близькі — модель не впевнена"
        elif margin < 0.20:
            conf_txt += "  ⚡ Близькі класи — дробні зміни можуть змінити результат"
        
        tk.Label(banner, text=conf_txt, bg=bg_c, fg=color, font=("Segoe UI",11)).pack(pady=(0,6))
        
        # Показуємо відхилення від норми здорової рослини — лише як довідкова інформація
        n_dev = res.get("n_deviations", 0)
        n_border = sum(1 for st, _ in statuses if st == "borderline")
        if n_dev == 0 and n_border == 0:
            dev_txt = "Усі показники в межах профілю здорової рослини"
        elif n_dev == 0:
            dev_txt = f"Відхилень від профілю здорової: 0  ·  на межі: {n_border}/8"
        elif n_border == 0:
            dev_txt = f"Відхилень від профілю здорової: {n_dev}/8"
        else:
            dev_txt = f"Відхилень від профілю здорової: {n_dev}/8  ·  на межі: {n_border}/8"
        tk.Label(banner, text=dev_txt, bg=bg_c, fg=color, font=("Segoe UI",9)).pack(pady=(0,4))
        feat_pct = int(round(res.get("feature_influence", 0.22) * 100))
        tk.Label(banner,
                 text=f"(клас: нейромережа + усі 8 показників; внесок ознак ~{feat_pct}%)",
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

        card_sens = self._card(self.result_frame,
                               "Показники біосенсорів (еталон здорової + внесок у класифікацію)")
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
            
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
            
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
        threading.Thread(target=self._predict_worker, args=(values,), daemon=True).start()
    
    def _predict_worker(self, values):
        """Окремий потік для передбачення без заморожування UI"""
        try:
            from models.predictor import get_probabilities
            X = np.array([values], dtype=np.float32)
            X_sc = self.scaler.transform(X)
            probs = get_probabilities(self.model, X_sc)[0]
            self.after(0, lambda: self._show_result(values, probs))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Помилка", str(e)))

    def _open_sensitivity_analysis(self):
        """Відкриває інтерактивне вікно аналізу чутливості обраної ознаки."""
        if not self._model_loaded:
            messagebox.showwarning("Модель не завантажена", "Спочатку навчіть або завантажте модель.")
            return
        try:
            current_values = [float(v.get()) for v in self.entries]
        except ValueError:
            messagebox.showerror("Помилка", "Перевірте значення — мають бути числами.")
            return

        win = tk.Toplevel(self)
        win.title("Аналіз чутливості моделі")
        win.geometry("800x680")
        win.configure(bg=BG)
        win.transient(self)
        win.grab_set()

        lbl = tk.Label(win, text="📊 Аналіз чутливості передбачень до змін ознак",
                       bg=BG, fg=ACCENT, font=("Segoe UI", 12, "bold"))
        lbl.pack(pady=10)

        sel_frame = tk.Frame(win, bg=BG)
        sel_frame.pack(fill="x", padx=20)
        tk.Label(sel_frame, text="Оберіть біосенсорний показник:", bg=BG, fg=TEXT, font=("Segoe UI", 10)).pack(side="left", padx=5)

        feat_combo = ttk.Combobox(sel_frame, values=FEATURE_UA, state="readonly", width=30)
        feat_combo.pack(side="left", padx=5)
        feat_combo.current(0)

        plot_frame = tk.Frame(win, bg=CARD, relief="flat", bd=1, highlightbackground=BORDER, highlightthickness=1)
        plot_frame.pack(fill="both", expand=True, padx=20, pady=(6, 0))

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        fig = Figure(figsize=(6.5, 4), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- Slider block ---
        slider_frame = tk.Frame(win, bg=BG)
        slider_frame.pack(fill="x", padx=20, pady=(4, 0))

        slider_lbl = tk.Label(slider_frame, text="Поточне значення:", bg=BG, fg=TEXT, font=("Segoe UI", 9))
        slider_lbl.pack(side="left", padx=(0, 6))

        slider_val_lbl = tk.Label(slider_frame, text="", bg=BG, fg=ACCENT, font=("Segoe UI", 9, "bold"), width=10, anchor="w")
        slider_val_lbl.pack(side="left")

        slider_var = tk.DoubleVar()
        slider = tk.Scale(
            slider_frame,
            variable=slider_var,
            orient="horizontal",
            showvalue=False,
            bg=BG, fg=TEXT, troughcolor=CARD,
            highlightthickness=0,
            bd=0,
        )
        slider.pack(side="left", fill="x", expand=True, padx=6)

        # Pre-computed curve cache (keyed by feat_idx)
        _cache = {}

        def _build_curves(feat_idx):
            """Compute probability curves for all x across the feature range."""
            lo, hi, _ = self.feature_ranges[feat_idx]
            x_vals = np.linspace(lo, hi, 200)

            test_inputs = np.array(
                [current_values[:feat_idx] + [x] + current_values[feat_idx + 1:] for x in x_vals],
                dtype=np.float32,
            )
            X_sc = self.scaler.transform(test_inputs)

            from models.predictor import get_probabilities
            mlp_probs = get_probabilities(self.model, X_sc)

            final_probs_list = []
            mlp_probs_list = []
            for inp_val, m_prob in zip(test_inputs, mlp_probs):
                res = classify(inp_val, m_prob, self.normal_ranges,
                               class_means=self.class_means, class_stds=self.class_stds)
                final_probs_list.append(res["final_probs"])
                mlp_probs_list.append(m_prob)

            return x_vals, np.array(final_probs_list), np.array(mlp_probs_list)

        def _redraw_vline(feat_idx, curr_val, x_vals, final_probs_arr):
            """Redraw only the vertical line — fast update on slider move."""
            ax.clear()
            lo, hi, _ = self.feature_ranges[feat_idx]

            for c_idx, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
                ax.plot(x_vals, final_probs_arr[:, c_idx] * 100, label=name, color=color, lw=2)

            ax.axvline(x=curr_val, color="red", linestyle="--", alpha=0.8,
                       label=f"Поточне ({_fmt_val(curr_val)})")
            ax.set_title(f"Вплив ознаки '{FEATURE_UA[feat_idx]}' на класифікацію",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel(FEATURE_UA[feat_idx], fontsize=9)
            ax.set_ylabel("Ймовірність класу (%)", fontsize=9)
            ax.set_ylim(-5, 105)
            ax.set_xlim(lo, hi)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.6)
            fig.tight_layout()
            canvas.draw_idle()

        def _fmt_val(v):
            if abs(v) >= 10 or v == 0:
                return f"{v:.2f}"
            return f"{v:.4f}"

        def on_slider_move(*_):
            feat_idx = feat_combo.current()
            if feat_idx < 0 or feat_idx not in _cache:
                return
            curr_val = slider_var.get()
            slider_val_lbl.config(text=_fmt_val(curr_val))
            x_vals, final_probs_arr, _ = _cache[feat_idx]
            _redraw_vline(feat_idx, curr_val, x_vals, final_probs_arr)

        slider_var.trace_add("write", on_slider_move)

        def update_plot(*args):
            feat_idx = feat_combo.current()
            if feat_idx < 0:
                return

            lo, hi, _ = self.feature_ranges[feat_idx]

            # Reconfigure slider for this feature
            slider.config(from_=lo, to=hi, resolution=(hi - lo) / 500)

            init_val = float(np.clip(current_values[feat_idx], lo, hi))
            slider_var.set(init_val)
            slider_val_lbl.config(text=_fmt_val(init_val))

            # Build / retrieve cached curves
            if feat_idx not in _cache:
                _cache[feat_idx] = _build_curves(feat_idx)

            x_vals, final_probs_arr, _ = _cache[feat_idx]
            _redraw_vline(feat_idx, init_val, x_vals, final_probs_arr)

        feat_combo.bind("<<ComboboxSelected>>", update_plot)

        # Info label
        info_lbl = tk.Label(win,
                            text="Перетягуйте слайдер — лінія рухається плавно без перерахунку кривих.",
                            bg=BG, fg="#888888", font=("Segoe UI", 8))
        info_lbl.pack(pady=(2, 8))

        update_plot()

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
        apply_smote = cfg.get("preprocessing", {}).get("apply_smote", True)

        card_data=self._card(top,"Датасет")
        row1=tk.Frame(card_data,bg=CARD); row1.pack(fill="x", pady=2)
        tk.Label(row1,text="CSV:",bg=CARD,fg=TEXT,font=("Segoe UI",9), width=12, anchor="w").pack(side="left")
        self.var_csv=tk.StringVar(value=train_path)
        tk.Entry(row1,textvariable=self.var_csv,width=52,font=("Segoe UI",9),relief="flat",
                 bg="#F0EFF8",highlightthickness=1,highlightbackground=BORDER).pack(side="left",padx=6)
        tk.Button(row1,text="Огляд…",bg=BG,fg=ACCENT,font=("Segoe UI",9),relief="flat",cursor="hand2",
                  command=self._browse_csv).pack(side="left")
        val_split = cfg.get("training", {}).get("validation_split", 0.2)

        card_hp=self._card(top,"Гіперпараметри"); hp_grid=tk.Frame(card_hp,bg=CARD); hp_grid.pack(fill="x")
        params=[
            ("Епохи:", epochs),
            ("Batch size:", batch_size),
            ("Learning rate:", lr),
            ("Patience:", patience),
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
            from preprocessing.splitter import split_from_config, format_split_summary, get_split_description
            from preprocessing.scaler import fit_transform, transform, save_scaler
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
            from visualization.plots import plot_training_history, plot_class_distribution, plot_smote_comparison
            from visualization.feature_imp import plot_correlation_heatmap
            from visualization.roc_curves import plot_roc_curves
            with open("config.yaml") as f: config=yaml.safe_load(f)
            from utils.seed import set_seed
            _seed = config.get("data", {}).get("random_state", 42)
            set_seed(_seed)
            self._log(f"  Seed зафіксовано: {_seed} (Python / NumPy / TensorFlow)", "green")
            config["training"]["epochs"]=int(self.hp_vars[0].get())
            config["training"]["batch_size"]=int(self.hp_vars[1].get())
            config["training"]["learning_rate"]=float(self.hp_vars[2].get())
            config["training"]["patience"]=int(self.hp_vars[3].get())
            config["data"]["path"]=csv_src
            config["preprocessing"]["apply_smote"]=self.var_smote.get()
            val_split = config.get("training", {}).get("validation_split", 0.2)
            config["data"]["split_strategy"] = get_split_description(val_split)
            with open("config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True)
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
            self._log("\n[2/8] Розбиття 80/20 train/test (sklearn train_test_split)...","yellow")
            X_train, X_test, y_train, y_test = split_from_config(X, y, config)
            split_info = format_split_summary(y_train, y_test, config)
            self._log(
                f"  Train: {split_info['train']} ({split_info['train_pct']:.1f}%)  "
                f"Test: {split_info['test']} ({split_info['test_pct']:.1f}%)",
                "green",
            )
            self._log(
                f"  Val на графіках: validation_split={split_info['val_split']} (з train, не окремий файл). Dropout=0 під час валідації → val_acc може бути вища за train_acc",
                "green",
            )
            self._log(f"  Стратегія: {split_info['strategy']}", "green")
            
            from preprocessing.augmentor import apply_smote
            from collections import Counter

            smote_stats = {"applied": False, "before": {}, "after": {}}
            smote_stats["before"] = {int(k): int(v) for k, v in Counter(y_train).items()}

            self._log("  Scaler fit — лише на train; SMOTE — лише на train","green")
            X_train_sc, scaler = fit_transform(X_train)
            X_test_sc = transform(X_test, scaler)
            save_scaler(scaler, "saved_models/scaler.pkl")
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
            
            X_train_for_model, y_train_for_model = X_train_sc, y_train
            if config["preprocessing"]["apply_smote"]:
                self._log("  Застосовуємо SMOTE до train даних...", "yellow")
                X_train_for_model, y_train_for_model = apply_smote(
                    X_train_sc, y_train, random_state=config["data"].get("random_state", 42))
                smote_stats["applied"] = True
                smote_stats["after"] = {int(k): int(v) for k, v in Counter(y_train_for_model).items()}
                try:
                    plot_smote_comparison(y_train, y_train_for_model, "results/smote_comparison.png")
                    self._log("  ✓ Графік SMOTE: results/smote_comparison.png", "green")
                except Exception as e:
                    self._log(f"  [!] Помилка SMOTE графіку: {e}", "red")
                self._log(f"  ✓ SMOTE: {X_train_for_model.shape[0]} зразків", "green")
                    
            history=train_model(mlp,X_train_for_model,y_train_for_model,None,None,config,"saved_models/mlp_best.keras",
                                custom_callbacks=[UICallback(self)])
            plot_training_history(history,"results/training_history_mlp.png")
            self._log("\n[4/8] Оцінювання на тестовій вибірці (20%)...","yellow")
            y_proba=mlp.predict(X_test_sc,verbose=0); y_pred=np.argmax(y_proba,axis=1)
            from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
            acc=accuracy_score(y_test,y_pred); f1=f1_score(y_test,y_pred,average="weighted")
            try: auc=roc_auc_score(y_test,y_proba,multi_class="ovr",average="weighted")
            except: auc=0.0
            self._log(f"\n  Test Accuracy: {acc:.4f}","green")
            self._log(f"  Test F1:       {f1:.4f}","green")
            self._log(f"  Test ROC-AUC:  {auc:.4f}","green")
            plot_confusion_matrix(y_test,y_pred,"results/confusion_matrix.png")
            roc_res=plot_roc_curves(y_test,y_proba,"results/roc_curves.png")
            self._log(f"  AUC micro={roc_res['auc_micro']:.4f}  macro={roc_res['auc_macro']:.4f}","green")
            err=analyze_errors(X_test_sc,y_test,y_pred,y_proba,save_dir="results")
            self._log(f"  Помилок: {err['n_errors']} ({err['error_rate']*100:.1f}%)","green")
            self._log("\n[5/8] Порівняння MLP vs CNN...","yellow")
            _mlp2=build_mlp(input_dim=X_train_sc.shape[1],hidden_layers=[128,64,32],dropout=0.3,
                            learning_rate=config["training"]["learning_rate"])
            _cnn=build_cnn(input_dim=X_train_sc.shape[1],learning_rate=config["training"]["learning_rate"])
            arch=compare_architectures({"MLP":_mlp2,"CNN":_cnn},X_train_sc,y_train,None,None,
                                       X_test_sc,y_test,config,save_dir="results")
            for aname,ares in arch.items():
                self._log(f"  {aname}: acc={ares['accuracy']:.4f}  f1={ares['f1_weighted']:.4f}","green")
            
            self._log("\n[7/8] Baseline порівняння...", "yellow")
            baseline=train_baselines(X_train_sc,y_train,X_test_sc,y_test,"saved_models")
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
            metrics_test=evaluate(y_test,y_pred,y_proba)
            
            save_report(metrics_test,cv,baseline,"results/report.json",
                extra={
                       "roc_curves":roc_res,"error_analysis":err,
                       "arch_comparison":{n:{k:v for k,v in r.items() if k not in ("history","y_pred","y_proba")}
                                          for n,r in arch.items()},
                       "split":{"train":len(y_train),"test":len(y_test),
                               "validation_split":config["training"].get("validation_split", 0.2),
                               "split_strategy":get_split_description(
                                   config["training"].get("validation_split", 0.2)),
                               "dataset":csv_src},
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
            self.scaler=scaler; self.model=mlp; self._model_loaded=True
            self._load_dataset_context()
            self.after(0, self._refresh_info_tab)
            self.lbl_model_status.configure(text="● модель завантажена",fg="#CCFFCC")
            messagebox.showinfo("Готово",f"Навчання завершено!\n\nTest Accuracy: {acc:.4f}\nTest F1: {f1:.4f}\nTest ROC-AUC: {auc:.4f}\nCV Accuracy: {cv['accuracy_mean']:.4f}±{cv['accuracy_std']:.4f}\n\nРезультати в папці results/")
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
        try:
            df=pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Помилка",f"Не вдається прочитати файл:\n{e}"); return
        missing=[c for c in FEATURE_COLS if c not in df.columns]
        if missing: messagebox.showerror("Помилка",f"Відсутні колонки:\n{', '.join(missing)}"); return
        threading.Thread(target=self._batch_worker, args=(df,), daemon=True).start()
    
    def _batch_worker(self, df):
        """Окремий потік для масового аналізу без заморожування UI"""
        try:
            from models.predictor import get_probabilities
            X = df[FEATURE_COLS].values.astype(np.float32)
            X_sc = self.scaler.transform(X)
            raw_proba = get_probabilities(self.model, X_sc)

            results = []
            counts = [0, 0, 0, 0]
            for i, (vals, raw_prob) in enumerate(zip(X, raw_proba)):
                res = classify(vals.tolist(), raw_prob, self.normal_ranges,
                               class_means=self.class_means, class_stds=self.class_stds)
                pred = res["final_cls"]
                conf = res["confidence"]
                results.append({
                    "index": i + 1, "values": vals, "pred": pred,
                    "confidence": conf, "name": CLASS_NAMES[pred],
                })
                counts[pred] += 1
            self.after(0, lambda: self._update_batch_ui(results, counts, len(X), df))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Помилка", f"Помилка аналізу:\n{str(e)}"))
    
    def _update_batch_ui(self,results,counts,total,df):
        """Оновлює UI з результатами аналізу"""
        for row in self.tree.get_children(): self.tree.delete(row)
        self._batch_results=[]
        for res in results:
            vals=res["values"]; pred=res["pred"]; conf=res["confidence"]; tag=f"class{pred}"
            row_data=(res["index"],)+tuple(f"{v:.2f}" for v in vals)+(CLASS_NAMES[pred],f"{conf:.1f}%")
            self.tree.insert("","end",values=row_data,tags=(tag,))
            self._batch_results.append({**{FEATURE_COLS[j]:vals[j] for j in range(8)},"predicted_class":pred,"predicted_name":CLASS_NAMES[pred],"confidence":conf})
        for w in self.batch_stats.winfo_children(): w.destroy()
        tk.Label(self.batch_stats,text=f"Всього: {total}",bg=BG,fg=TEXT,font=("Segoe UI",10,"bold")).pack(side="left",padx=(0,20))
        for i,(name,cnt) in enumerate(zip(CLASS_NAMES,counts)):
            tk.Label(self.batch_stats,text=f"{name}: {cnt} ({cnt/total*100:.0f}%)",bg=CLASS_BG[i],fg=CLASS_COLORS[i],font=("Segoe UI",9,"bold"),padx=8,pady=3).pack(side="left",padx=4)
        if "plant_health_status" in df.columns:
            from sklearn.metrics import accuracy_score
            preds=np.array([r["pred"] for r in results])
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