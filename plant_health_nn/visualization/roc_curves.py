import matplotlib.pyplot as plt, numpy as np, os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from preprocessing.loader import CLASS_NAMES
COLORS = ["#E24B4A","#EF9F27","#639922","#1D9E75"]

def plot_roc_curves(y_true, y_proba, save_path="results/roc_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    classes = sorted(CLASS_NAMES.keys()); y_bin = label_binarize(y_true, classes=classes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]; aucs = []
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_proba[:,i])
        ra = auc(fpr, tpr); aucs.append(ra)
        ax.plot(fpr, tpr, color=COLORS[i], lw=2, label=f"{CLASS_NAMES[cls]} (AUC = {ra:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Випадковий класифікатор")
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02]); ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)"); ax.set_title("ROC-криві по класах", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    auc_m = auc(fpr_m, tpr_m); ax2.plot(fpr_m, tpr_m, color="#534AB7", lw=2.5, label=f"Мікро-avg (AUC = {auc_m:.3f})")
    all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:,i], y_proba[:,i])[0] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        fp, tp, _ = roc_curve(y_bin[:,i], y_proba[:,i]); mean_tpr += np.interp(all_fpr, fp, tp)
    mean_tpr /= len(classes); auc_mac = auc(all_fpr, mean_tpr)
    ax2.plot(all_fpr, mean_tpr, color="#D85A30", lw=2.5, linestyle="--", label=f"Макро-avg (AUC = {auc_mac:.3f})")
    for i, cls in enumerate(classes):
        fp, tp, _ = roc_curve(y_bin[:,i], y_proba[:,i]); ax2.plot(fp, tp, color=COLORS[i], lw=1, alpha=0.35)
    ax2.plot([0,1],[0,1],"k--",lw=1,alpha=0.5); ax2.set_xlim([0,1]); ax2.set_ylim([0,1.02])
    ax2.set_xlabel("False Positive Rate (FPR)"); ax2.set_ylabel("True Positive Rate (TPR)")
    ax2.set_title("ROC — усереднені криві", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9); ax2.grid(True, alpha=0.3)
    plt.suptitle("ROC-аналіз нейромережевого класифікатора стану рослин", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"ROC-криві збережено: {save_path}")
    return {"auc_per_class": {CLASS_NAMES[c]: aucs[i] for i,c in enumerate(classes)},
            "auc_micro": float(auc_m), "auc_macro": float(auc_mac)}
