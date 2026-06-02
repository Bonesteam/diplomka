import numpy as np, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec, os
from preprocessing.loader import CLASS_NAMES, FEATURE_COLS
COLORS = ["#E24B4A","#EF9F27","#639922","#1D9E75"]
FEATURE_UA = ["Флуоресценція","Колоримет.","SPR-сигнал","Темп. листа","Хлорофіл","Вологість","Поглинання","VOC"]

def analyze_errors(X_test, y_test, y_pred, y_proba, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    mask = y_pred != y_test
    X_err, y_true_e, y_pred_e, y_prob_e = X_test[mask], y_test[mask], y_pred[mask], y_proba[mask]
    n_errors = mask.sum()
    print(f"  Помилок: {n_errors} з {len(y_test)} ({n_errors/len(y_test)*100:.1f}%)")
    pairs = {}
    for t, p in zip(y_true_e, y_pred_e):
        k = (int(t), int(p)); pairs[k] = pairs.get(k, 0) + 1

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    classes = sorted(CLASS_NAMES.keys())

    ax1 = fig.add_subplot(gs[0, 0])
    err_by = [np.sum(y_true_e == c) for c in classes]
    tot_by = [np.sum(y_test == c) for c in classes]
    rate   = [e/t if t > 0 else 0 for e, t in zip(err_by, tot_by)]
    bars = ax1.bar([CLASS_NAMES[c] for c in classes], rate, color=COLORS, alpha=0.85)
    for bar, r, e, t in zip(bars, rate, err_by, tot_by):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{e}/{t}\n({r*100:.0f}%)", ha="center", fontsize=8.5)
    ax1.set_ylabel("Частка помилок"); ax1.set_title("Частота помилок по класах", fontweight="bold")
    ax1.set_ylim(0, max(rate)*1.5+0.05); plt.setp(ax1.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    if pairs:
        pl = [f"{CLASS_NAMES[k[0]][:8]}\u2192{CLASS_NAMES[k[1]][:8]}" for k in pairs]
        pv = list(pairs.values()); pc = [COLORS[k[0]] for k in pairs]
        si = np.argsort(pv)[::-1]
        ax2.barh([pl[i] for i in si], [pv[i] for i in si], color=[pc[i] for i in si], alpha=0.8)
    ax2.set_xlabel("Кількість помилок"); ax2.set_title("Типи помилок (true → predicted)", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    conf_err = np.max(y_prob_e, axis=1); conf_ok = np.max(y_proba[y_pred==y_test], axis=1)
    ax3.hist(conf_ok,  bins=20, alpha=0.65, color="#1D9E75", label=f"Правильні (n={len(conf_ok)})",  density=True)
    ax3.hist(conf_err, bins=20, alpha=0.65, color="#E24B4A", label=f"Помилки (n={len(conf_err)})", density=True)
    ax3.axvline(np.mean(conf_ok),  color="#1D9E75", lw=2, linestyle="--", label=f"Сер. правильні: {np.mean(conf_ok):.2f}")
    ax3.axvline(np.mean(conf_err), color="#E24B4A", lw=2, linestyle="--", label=f"Сер. помилки: {np.mean(conf_err):.2f}")
    ax3.set_xlabel("Впевненість моделі"); ax3.set_ylabel("Щільність")
    ax3.set_title("Розподіл впевненості", fontweight="bold"); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, :2])
    X_ok = X_test[y_pred==y_test]
    mc, me = X_ok.mean(axis=0), (X_err.mean(axis=0) if len(X_err)>0 else np.zeros(len(FEATURE_COLS)))
    se = X_err.std(axis=0) if len(X_err)>0 else np.zeros(len(FEATURE_COLS))
    xx = np.arange(len(FEATURE_COLS)); w = 0.35
    ax4.bar(xx-w/2, mc, w, label="Правильно класифіковані", color="#1D9E75", alpha=0.75)
    ax4.bar(xx+w/2, me, w, label="Помилково класифіковані", color="#E24B4A", alpha=0.75, yerr=se, capsize=3)
    ax4.set_xticks(xx); ax4.set_xticklabels(FEATURE_UA, rotation=30, ha="right", fontsize=9)
    ax4.set_ylabel("Середнє значення (норм.)"); ax4.set_title("Середні значення ознак: правильні vs помилки", fontweight="bold")
    ax4.legend(); ax4.grid(axis="y", alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    if len(y_prob_e) > 0:
        hi = np.argsort(conf_err)[:min(8, len(conf_err))]
        ht = [CLASS_NAMES[y_true_e[i]] for i in hi]; hp = [CLASS_NAMES[y_pred_e[i]] for i in hi]
        hc = [conf_err[i]*100 for i in hi]; yp = np.arange(len(hi))
        bars5 = ax5.barh(yp, hc, color="#EF9F27", alpha=0.8)
        ax5.set_yticks(yp); ax5.set_yticklabels([f"{t[:8]}\u2192{p[:8]}" for t,p in zip(ht,hp)], fontsize=8)
        ax5.set_xlabel("Впевненість (%)"); ax5.set_title("Найважчі помилки\n(найнижча впевненість)", fontweight="bold")
        ax5.set_xlim(0, 100)
        for bar, c in zip(bars5, hc):
            ax5.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f"{c:.1f}%", va="center", fontsize=8)
        ax5.grid(axis="x", alpha=0.3)

    plt.suptitle("Детальний аналіз помилок класифікатора", fontsize=15, fontweight="bold", y=1.01)
    path = os.path.join(save_dir, "error_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Збережено: {path}")
    return {"n_errors": int(n_errors), "error_rate": float(n_errors/len(y_test)),
            "confusion_pairs": {f"{CLASS_NAMES[k[0]]}\u2192{CLASS_NAMES[k[1]]}": v for k, v in pairs.items()}}
