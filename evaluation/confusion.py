import matplotlib.pyplot as plt, seaborn as sns, numpy as np, os
from sklearn.metrics import confusion_matrix
from preprocessing.loader import CLASS_NAMES

def plot_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in zip(axes, [cm, cm_norm],
        ["Матриця плутанини (абс.)", "Матриця плутанини (норм.)"], ["d", ".2f"]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title); ax.set_ylabel("Справжній клас"); ax.set_xlabel("Передбачений клас")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Збережено: {save_path}")
