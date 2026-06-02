import numpy as np, os, tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_model(model, X_train, y_train, X_val, y_val, config, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    classes = np.unique(y_train)
    total = len(y_train)
    class_weight = {c: total / (len(classes) * np.sum(y_train == c)) for c in classes}
    
    has_val = X_val is not None and y_val is not None
    monitor_loss = "val_loss" if has_val else "loss"
    monitor_acc = "val_accuracy" if has_val else "accuracy"
    
    callbacks = [
        EarlyStopping(monitor=monitor_loss, patience=config["training"]["patience"],
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(save_path, monitor=monitor_acc, save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor=monitor_loss, factor=0.5, patience=7, verbose=1),
    ]
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val) if has_val else None,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        class_weight=class_weight, callbacks=callbacks, verbose=1)
    return history