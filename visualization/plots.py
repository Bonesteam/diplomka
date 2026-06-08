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

def plot_smote_comparison(y_before, y_after, save_path="results/smote_comparison.png"):
    """Порівняння розподілу класів до та після SMOTE."""
    from preprocessing.loader import CLASS_NAMES
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    unique_classes = sorted(set(y_before.tolist()))
    labels = [CLASS_NAMES[i] for i in unique_classes]
    counts_before = [np.sum(y_before == i) for i in unique_classes]
    counts_after = [np.sum(y_after == i) for i in unique_classes]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = sns.color_palette("husl", len(labels))
    
    # До SMOTE
    bars1 = axes[0].bar(labels, counts_before, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_title("До SMOTE", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Кількість зразків")
    axes[0].tick_params(axis='x', rotation=20)
    for bar, cnt in zip(bars1, counts_before):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, str(cnt), 
                     ha="center", va="bottom", fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)
    
    # Після SMOTE
    bars2 = axes[1].bar(labels, counts_after, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_title("Після SMOTE", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Кількість зразків")
    axes[1].tick_params(axis='x', rotation=20)
    for bar, cnt in zip(bars2, counts_after):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, str(cnt), 
                     ha="center", va="bottom", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)
    
    # Вирівняємо масштаб осей Y
    max_val = max(max(counts_before), max(counts_after))
    axes[0].set_ylim(0, max_val * 1.15)
    axes[1].set_ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Збережено: {save_path}")
