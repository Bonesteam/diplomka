import matplotlib.pyplot as plt, numpy as np, os
from preprocessing.loader import FEATURE_COLS

def plot_feature_importance_rf(rf_model, save_path="results/feature_importance.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imp = rf_model.feature_importances_; idx = np.argsort(imp)[::-1]
    sf = [FEATURE_COLS[i] for i in idx]; si = imp[idx]
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(sf)), si, color=plt.cm.RdYlGn(si/si.max()))
    plt.xticks(range(len(sf)), sf, rotation=35, ha="right"); plt.ylabel("Важливість ознаки")
    plt.title("Важливість біосенсорних ознак (Random Forest)"); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Збережено: {save_path}")

def plot_correlation_heatmap(X, save_path="results/correlation.png"):
    import pandas as pd, seaborn as sns
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    corr = pd.DataFrame(X, columns=FEATURE_COLS).corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.title("Кореляційна матриця ознак"); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Збережено: {save_path}")
