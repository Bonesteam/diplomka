import matplotlib.pyplot as plt
import os


def plot_correlation_heatmap(X, save_path="results/correlation.png"):
    import pandas as pd
    import seaborn as sns
    from preprocessing.loader import FEATURE_COLS

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    corr = pd.DataFrame(X, columns=FEATURE_COLS).corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.title("Кореляційна матриця ознак")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Збережено: {save_path}")
