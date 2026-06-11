import numpy as np, matplotlib.pyplot as plt, os, time
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

def compare_architectures(models_dict, X_train, y_train, X_val, y_val, X_test, y_test, config, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    for name, model in models_dict.items():
        print(f"  Навчання: {name}")
        Xtr = X_train[:,:,np.newaxis] if "CNN" in name else X_train
        Xv  = X_val[:,:,np.newaxis]   if "CNN" in name and X_val is not None else X_val
        Xte = X_test[:,:,np.newaxis]  if "CNN" in name else X_test
        classes = np.unique(y_train); total = len(y_train)
        cw = {c: total/(len(classes)*np.sum(y_train==c)) for c in classes}
        t0 = time.time()
        
        has_explicit_val = Xv is not None and y_val is not None
        val_split = config.get("training", {}).get("validation_split", 0.2)
        use_val_split = not has_explicit_val and val_split and val_split > 0
        has_val = has_explicit_val or use_val_split
        monitor_loss = "val_loss" if has_val else "loss"

        fit_kwargs = dict(
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            class_weight=cw,
            callbacks=[
                EarlyStopping(monitor=monitor_loss, patience=config["training"]["patience"],
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor=monitor_loss, factor=0.5, patience=7, verbose=0),
            ],
            verbose=0,
        )
        if has_explicit_val:
            fit_kwargs["validation_data"] = (Xv, y_val)
        elif use_val_split:
            fit_kwargs["validation_split"] = val_split
        history = model.fit(Xtr, y_train, **fit_kwargs)
        train_time = time.time() - t0
        y_proba = model.predict(Xte, verbose=0); y_pred = np.argmax(y_proba, axis=1)
        acc = accuracy_score(y_test, y_pred); f1w = f1_score(y_test, y_pred, average="weighted")
        f1m = f1_score(y_test, y_pred, average="macro")
        try:
            yb = label_binarize(y_test, classes=sorted(np.unique(y_test)))
            auc = roc_auc_score(yb, y_proba, multi_class="ovr", average="weighted")
        except Exception: auc = 0.0
        results[name] = {"accuracy": acc, "f1_weighted": f1w, "f1_macro": f1m, "roc_auc": auc,
                         "train_time": train_time, "n_params": model.count_params(),
                         "n_epochs": len(history.history["loss"]), "history": history.history,
                         "y_pred": y_pred, "y_proba": y_proba}
        print(f"    acc={acc:.4f}  f1={f1w:.4f}  auc={auc:.4f}  t={train_time:.1f}s")
    _plot(results, save_dir); return results

def _plot(results, save_dir):
    names = list(results.keys()); colors = ["#534AB7","#D85A30","#1D9E75","#BA7517"]
    metrics = ["accuracy","f1_weighted","f1_macro","roc_auc"]
    labels  = ["Accuracy","F1 (weighted)","F1 (macro)","ROC-AUC"]
    x = np.arange(len(metrics)); w = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    for i, (name, res) in enumerate(results.items()):
        vals = [res[m] for m in metrics]; offset = (i-len(names)/2+0.5)*w
        bars = ax.bar(x+offset, vals, w, label=name, color=colors[i%len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10); ax.set_ylim(0, 1.12)
    ax.set_ylabel("Значення метрики"); ax.set_title("Порівняння метрик якості", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax2 = axes[1]
    for i, (name, res) in enumerate(results.items()):
        va = res["history"].get("val_accuracy", res["history"].get("accuracy", []))
        ax2.plot(va, label=name, color=colors[i%len(colors)], lw=2)
    ax2.set_xlabel("Епоха")
    has_val = "val_accuracy" in results[names[0]]["history"]
    ax2.set_ylabel("Val Accuracy" if has_val else "Accuracy")
    ax2.set_title("Криві навчання (val accuracy)" if has_val else "Криві навчання (accuracy)", fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax3 = axes[2]
    times = [results[n]["train_time"] for n in names]; params = [results[n]["n_params"]/1000 for n in names]
    bars1 = ax3.bar(names, times, color=[colors[i] for i in range(len(names))], alpha=0.7, label="Час (с)")
    ax3.set_ylabel("Час навчання (с)", color="#333")
    ax3b = ax3.twinx(); ax3b.plot(names, params, "D--", color="#888", lw=1.5, ms=8, label="Параметри (тис.)")
    ax3b.set_ylabel("Параметри (тис.)", color="#888")
    for bar, t in zip(bars1, times):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{t:.1f}с", ha="center", fontsize=9)
    ax3.set_title("Час навчання та складність", fontweight="bold")
    l1, lb1 = ax3.get_legend_handles_labels(); l2, lb2 = ax3b.get_legend_handles_labels()
    ax3.legend(l1+l2, lb1+lb2, fontsize=9); ax3.grid(axis="y", alpha=0.3)
    plt.suptitle("Порівняльний аналіз архітектур нейронних мереж", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "arch_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Збережено: {path}")
    print("\n"+"="*65)
    print(f"{'Модель':<12} {'Accuracy':>10} {'F1(w)':>8} {'AUC':>8} {'Час(с)':>8} {'Параметри':>12}")
    print("-"*65)
    for name, res in results.items():
        print(f"{name:<12} {res['accuracy']:>10.4f} {res['f1_weighted']:>8.4f} {res['roc_auc']:>8.4f} {res['train_time']:>8.1f} {res['n_params']:>12,}")
    print("="*65)
