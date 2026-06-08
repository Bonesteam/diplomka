"""Коротке тестове навчання (3 епохи) — 80/20 + validation_split як у Kiril."""
import os
import sys
import yaml
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    config["training"]["epochs"] = 3
    config["training"]["patience"] = 5
    config["preprocessing"]["apply_smote"] = False

    from preprocessing.loader import load_data
    from preprocessing.splitter import split_from_config, format_split_summary
    from preprocessing.scaler import fit_transform, transform
    from models.mlp_model import build_mlp
    from models.trainer import train_model
    from visualization.plots import plot_training_history
    from sklearn.metrics import accuracy_score

    csv_path = config["data"]["path"]
    print("=" * 55)
    print("  Test: 80/20 + validation_split (Kiril)")
    print("=" * 55)

    X, y, _ = load_data(csv_path)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = split_from_config(X, y, config)
    info = format_split_summary(y_train, y_test, config)
    print(f"  Train: {info['train']} ({info['train_pct']:.1f}%)")
    print(f"  Test:  {info['test']} ({info['test_pct']:.1f}%)")
    print(f"  Val on graphs: validation_split={info['val_split']}")

    X_train_sc, scaler = fit_transform(X_train)
    X_test_sc = transform(X_test, scaler)

    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    mlp = build_mlp(
        input_dim=X_train_sc.shape[1],
        hidden_layers=[128, 64, 32],
        dropout=0.3,
        learning_rate=config["training"]["learning_rate"],
    )

    print("\n  Training MLP (validation_split=0.2)...")
    history = train_model(
        mlp, X_train_sc, y_train, None, None, config,
        "saved_models/mlp_test.keras",
    )

    hist = history.history
    has_val = "val_loss" in hist and "val_accuracy" in hist
    print(f"\n  val_loss in history:     {'YES' if has_val else 'NO'}")
    print(f"  val_accuracy in history: {'YES' if has_val else 'NO'}")

    if has_val:
        for i in range(len(hist["loss"])):
            print(
                f"  Epoch {i+1}: loss={hist['loss'][i]:.4f} acc={hist['accuracy'][i]:.4f} "
                f"val_loss={hist['val_loss'][i]:.4f} val_acc={hist['val_accuracy'][i]:.4f}"
            )

    plot_path = "results/training_history_test.png"
    plot_training_history(history, plot_path)
    print(f"\n  Graph saved: {plot_path}")

    y_pred = np.argmax(mlp.predict(X_test_sc, verbose=0), axis=1)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {test_acc:.4f}")

    if not has_val or not os.path.exists(plot_path):
        sys.exit(1)

    print("\n  OK: 80/20 split + val curves work.")
    print("=" * 55)


if __name__ == "__main__":
    main()
