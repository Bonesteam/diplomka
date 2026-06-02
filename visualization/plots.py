import matplotlib.pyplot as plt, seaborn as sns, numpy as np, os

def plot_training_history(history, save_path="results/training_history_mlp.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Функція втрат"); axes[0].set_xlabel("Епоха"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.history["accuracy"], label="Train")
    if "val_accuracy" in history.history:
        axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set_title("Точність"); axes[1].set_xlabel("Епоха"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Збережено: {save_path}")

def plot_class_distribution(y, title="Розподіл класів", save_path="results/class_dist.png"):
    from preprocessing.loader import CLASS_NAMES
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = [CLASS_NAMES[i] for i in sorted(set(y.tolist()))]
    counts = [np.sum(y == i) for i in sorted(set(y.tolist()))]
    colors = sns.color_palette("husl", len(labels))
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(cnt), ha="center", va="bottom", fontsize=10)
    plt.title(title); plt.ylabel("Кількість зразків"); plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Збережено: {save_path}")

def plot_comparison_bar(metrics_nn, baseline_results, save_path="results/comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names = ["MLP (нейромережа)"] + list(baseline_results.keys())
    accs = [metrics_nn["accuracy"]] + [v["accuracy"] for v in baseline_results.values()]
    f1s  = [metrics_nn["f1_weighted"]] + [v["f1_weighted"] for v in baseline_results.values()]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x-w/2, accs, w, label="Accuracy", color="#4C72B0")
    ax.bar(x+w/2, f1s,  w, label="F1 (weighted)", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Значення метрики"); ax.set_title("Порівняння методів класифікації")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i-w/2, a+0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i+w/2, f+0.01, f"{f:.3f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Збережено: {save_path}")
