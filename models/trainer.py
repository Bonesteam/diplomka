"""Модуль тренування нейронних мереж з обробкою дисбалансу класів."""
import numpy as np, os, tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_model(model, X_train, y_train, X_val, y_val, config, save_path, custom_callbacks=None):
    """
    Тренує модель з урахуванням дисбалансу класів та оптимальною стратегією зупинки.
    Якщо X_val/y_val не передані — val береться з train через validation_split (як у Kiril).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    classes = np.unique(y_train)
    total = len(y_train)
    
    base_weights = {c: total / (len(classes) * np.sum(y_train == c)) for c in classes}
    class_weight = {}
    for c in classes:
        weight = base_weights[c]
        if np.sum(y_train == c) < 100:
            weight *= 2.0
        class_weight[c] = weight
    
    print(f"Class weights: {class_weight}")
    
    has_explicit_val = X_val is not None and y_val is not None
    val_split = config.get("training", {}).get("validation_split", 0.2)
    use_val_split = not has_explicit_val and val_split and val_split > 0
    has_val = has_explicit_val or use_val_split
    monitor_loss = "val_loss" if has_val else "loss"
    monitor_acc = "val_accuracy" if has_val else "accuracy"
    
    callbacks = [
        EarlyStopping(monitor=monitor_loss, patience=config["training"]["patience"],
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(save_path, monitor=monitor_acc, save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor=monitor_loss, factor=0.5,
                          patience=config["training"].get("reduce_patience", 7), verbose=1),
    ]
    if custom_callbacks:
        callbacks.extend(custom_callbacks)

    fit_kwargs = {
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "class_weight": class_weight,
        "callbacks": callbacks,
        "verbose": 1,
    }
    if has_explicit_val:
        fit_kwargs["validation_data"] = (X_val, y_val)
    elif use_val_split:
        fit_kwargs["validation_split"] = val_split

    history = model.fit(X_train, y_train, **fit_kwargs)
    return history
