import matplotlib.pyplot as plt, numpy as np, os, sklearn
from sklearn.manifold import TSNE
from preprocessing.loader import CLASS_NAMES

def plot_tsne(X, y, save_path="results/tsne.png", perplexity=30):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("Запуск t-SNE...")
    kw = dict(n_components=2, perplexity=perplexity, random_state=42)
    sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    kw["max_iter" if sk_ver >= (1,3) else "n_iter"] = 1000
    X_2d = TSNE(**kw).fit_transform(X)
    labels = sorted(CLASS_NAMES.keys()); colors = plt.cm.tab10(np.linspace(0, 0.6, len(labels)))
    plt.figure(figsize=(8, 6))
    for cls, color in zip(labels, colors):
        mask = y == cls
        plt.scatter(X_2d[mask,0], X_2d[mask,1], c=[color], label=CLASS_NAMES[cls], alpha=0.7, s=30, edgecolors="none")
    plt.legend(title="Стан рослини", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.title("t-SNE візуалізація біосенсорних даних"); plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Збережено: {save_path}")
