#!/usr/bin/env python3
"""Run ablation experiments: basic / +SMOTE / +class_weights / full.
Saves results to results/ablation.json and results/ablation_table.csv

Usage:
    python run_ablation.py [--dry-run] [--epochs N]
"""
import os, json, argparse, copy
import csv
import numpy as np

from utils.config_loader import load_config, ensure_dirs
from preprocessing.loader import load_data
from preprocessing.splitter import split_data
from preprocessing.scaler import fit_transform, transform, save_scaler
from preprocessing.augmentor import apply_smote
from evaluation.metrics import evaluate

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def build_mlp_custom(input_dim, num_classes=4, hidden_layers=None, dropout=0.3, l2_reg=1e-4, learning_rate=0.001):
    if hidden_layers is None:
        hidden_layers = [128, 64, 32]
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for units in hidden_layers:
        if l2_reg and l2_reg > 0:
            x = layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
        else:
            x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out, name="MLP_Ablation")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def compute_class_weight(y):
    classes = np.unique(y)
    total = len(y)
    cw = {int(c): float(total / (len(classes) * np.sum(y == c))) for c in classes}
    return cw


def run_experiment(name, cfg, config, dry_run=False, epochs_override=None):
    print(f"\n=== Experiment: {name} ===")
    # Load & split
    test_path = config["data"].get("test_path")
    has_test = False
    if test_path and os.path.exists(test_path):
        try:
            X_test, y_test, _ = load_data(test_path)
            has_test = True
        except Exception as e:
            print(f"Error loading test file: {e}")

    X, y, _ = load_data(config["data"]["path"]) if os.path.exists(config["data"]["path"]) else load_data("data/plant_health_biosensor_15k.csv")
    
    from sklearn.model_selection import train_test_split
    if has_test:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=config["data"].get("val_size", 0.2), random_state=config["data"]["random_state"], stratify=y
        )
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=config["data"]["test_size"], val_size=config["data"]["val_size"], random_state=config["data"]["random_state"]
        )

    X_train_sc, scaler = fit_transform(X_train, config["preprocessing"]["scaler"])
    X_val_sc = transform(X_val, scaler)
    X_test_sc = transform(X_test, scaler)
    # apply SMOTE if requested in cfg
    if cfg.get("apply_smote"):
        X_train_sc, y_train = apply_smote(X_train_sc, y_train, random_state=config["data"]["random_state"])
    # build model
    l2 = cfg.get("l2_reg", 1e-4)
    dropout = cfg.get("dropout", config["model"].get("dropout", 0.3))
    model = build_mlp_custom(X_train_sc.shape[1], num_classes=len(np.unique(y)),
                             hidden_layers=config["model"].get("hidden_layers"),
                             dropout=dropout, l2_reg=l2,
                             learning_rate=config["training"].get("learning_rate", 0.001))
    # callbacks and save path
    save_dir = config["paths"]["results"]
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(config["paths"]["saved_models"], f"ablation_{name}.keras")
    callbacks = [
        EarlyStopping(monitor=("val_loss" if X_val_sc is not None and y_val is not None else "loss"), patience=config["training"]["patience"], restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor=("val_loss" if X_val_sc is not None and y_val is not None else "loss"), factor=0.5, patience=7, verbose=1),
        ModelCheckpoint(model_path, monitor=("val_accuracy" if X_val_sc is not None and y_val is not None else "accuracy"), save_best_only=True, verbose=0)
    ]
    # class_weight
    class_weight = compute_class_weight(y_train) if cfg.get("use_class_weight") else None
    # epochs
    epochs = epochs_override if epochs_override is not None else config["training"]["epochs"]
    print(f"  Samples train={len(y_train)} test={len(y_test)} epochs={epochs} smote={cfg.get('apply_smote')} class_weight={bool(class_weight)} l2={l2} dropout={dropout}")
    if dry_run:
        print("  Dry run: skipping training")
        return {"name": name, "train_samples": int(len(y_train)), "test_samples": int(len(y_test)), "epochs": epochs}
    history = model.fit(X_train_sc, y_train, validation_data=(X_val_sc, y_val) if X_val_sc is not None and y_val is not None else None,
                        epochs=epochs, batch_size=config["training"]["batch_size"], callbacks=callbacks,
                        class_weight=class_weight, verbose=1)
    # evaluate
    y_proba = model.predict(X_test_sc, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = evaluate(y_test, y_pred, y_proba)
    # save scaler and model path already saved by checkpoint
    save_scaler(scaler, os.path.join(config["paths"]["saved_models"], f"scaler_ablation_{name}.pkl"))
    return {"name": name, "metrics": metrics, "history_len": len(history.history.get("loss", []))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Do not train, only print plan")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs per experiment")
    args = parser.parse_args()

    config = load_config("config.yaml")
    ensure_dirs({"paths": config["paths"], **config}) if isinstance(config, dict) else None

    # experiments
    experiments = [
        ("mlp_basic", {"apply_smote": False, "use_class_weight": False, "l2_reg": 0.0, "dropout": 0.0}),
        ("mlp_smote", {"apply_smote": True, "use_class_weight": False, "l2_reg": 0.0, "dropout": 0.0}),
        ("mlp_class_weights", {"apply_smote": False, "use_class_weight": True, "l2_reg": 0.0, "dropout": 0.0}),
        ("mlp_full", {"apply_smote": True, "use_class_weight": True, "l2_reg": 1e-4, "dropout": config["model"].get("dropout", 0.3)}),
    ]

    results = []
    for name, exp_cfg in experiments:
        res = run_experiment(name, exp_cfg, config, dry_run=args.dry_run, epochs_override=args.epochs)
        results.append(res)
        # write intermediate results
        with open(os.path.join(config["paths"]["results"], "ablation_partial.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # write final results
    out_path = os.path.join(config["paths"]["results"], "ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved ablation results to: {out_path}")

    # also write a CSV table for easy inclusion in reports
    csv_path = os.path.join(config["paths"]["results"], "ablation_table.csv")
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as cf:
            writer = None
            for r in results:
                name = r.get("name")
                metrics = r.get("metrics") or {}
                row = {
                    "Конфігурація": name,
                    "Accuracy": metrics.get("accuracy"),
                    "F1 (weighted)": metrics.get("f1_weighted"),
                    "F1 (macro)": metrics.get("f1_macro"),
                    "ROC-AUC": metrics.get("roc_auc")
                }
                if writer is None:
                    writer = csv.DictWriter(cf, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
        print(f"Saved ablation table CSV to: {csv_path}")
    except Exception:
        pass

    # merge ablation results into results/report.json if present (or create)
    report_path = os.path.join(config["paths"]["results"], "report.json")
    try:
        report = {}
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as rf:
                report = json.load(rf)
        # build experiments summary
        experiments = []
        for r in results:
            name = r.get("name")
            m = r.get("metrics") or {}
            experiments.append({
                "name": name,
                "accuracy": float(m.get("accuracy")) if m.get("accuracy") is not None else None,
                "f1_weighted": float(m.get("f1_weighted")) if m.get("f1_weighted") is not None else None,
                "f1_macro": float(m.get("f1_macro")) if m.get("f1_macro") is not None else None,
                "roc_auc": float(m.get("roc_auc")) if m.get("roc_auc") is not None else None,
            })
        report.setdefault("ablation", {})
        report["ablation"]["experiments"] = experiments
        report["ablation"]["csv"] = os.path.relpath(csv_path).replace('\\', '/')
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump(report, rf, ensure_ascii=False, indent=2)
        print(f"Merged ablation into report: {report_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
