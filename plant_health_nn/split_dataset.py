"""Розбиття одного CSV: 80% train / 20% test (sklearn train_test_split)."""
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing.loader import load_data
from preprocessing.splitter import split_from_config, format_split_summary


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    csv_path = config["data"]["path"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Датасет не знайдено: {csv_path}")

    df = pd.read_csv(csv_path)
    X, y, _ = load_data(csv_path)
    X_train, X_test, y_train, y_test = split_from_config(X, y, config)
    info = format_split_summary(y_train, y_test, config)

    rs = config["data"]["random_state"]
    test_size = config["data"]["test_size"]
    idx = df.index.to_numpy()
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=rs, stratify=y)

    os.makedirs("data", exist_ok=True)
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    df.loc[train_idx].reset_index(drop=True).to_csv(train_path, index=False)
    df.loc[test_idx].reset_index(drop=True).to_csv(test_path, index=False)

    print("=" * 55)
    print(f"  Train: {info['train']} ({info['train_pct']:.1f}%) -> {train_path}")
    print(f"  Test:  {info['test']} ({info['test_pct']:.1f}%) -> {test_path}")
    print(f"  Val on graphs: validation_split={config['training'].get('validation_split', 0.2)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
