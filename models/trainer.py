"""Модуль тренування нейронних мереж з обробкою дисбалансу класів."""
import numpy as np, os, tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_model(model, X_train, y_train, X_val, y_val, config, save_path, custom_callbacks=None):
    """
    Тренує модель з урахуванням дисбалансу класів та оптимальною стратегією зупинки.
    
    Args:
        model: Keras модель для тренування
        X_train, y_train: Дані та мітки для тренування
        X_val, y_val: Дані та мітки для валідації
        config (dict): Словник конфігурації з параметрами тренування
        save_path (str): Шлях для збереження найкращої моделі
        custom_callbacks (list): Додаткові callback'и (наприклад, для UI)
    
    Returns:
        history: Об'єкт History з метриками навчання кожної епохи
    
    Особливості:
        - Автоматичний розрахунок class_weight для боротьби з дисбалансом
        - 2x множник для рідких класів (< 100 зразків)
        - Early stopping, Learning rate scheduling, Model checkpoint
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    classes = np.unique(y_train)
    total = len(y_train)
    
    # Обчислюємо базові ваги обернено до частоти класу
    base_weights = {c: total / (len(classes) * np.sum(y_train == c)) for c in classes}
    
    # Підсилюємо ваги для малих класів (більше штрафу за помилки)
    # Клас 2 (Легкий стрес) має найменше зразків, тому повинен мати більшу вагу
    class_weight = {}
    for c in classes:
        weight = base_weights[c]
        # Додаємо множник: якщо клас рідкісний, увеличиваємо його вагу
        if np.sum(y_train == c) < 100:  # Якщо менше 100 зразків
            weight *= 2.0  # Збільшуємо в 2 рази
        class_weight[c] = weight
    
    print(f"Class weights: {class_weight}")
    
    has_val = X_val is not None and y_val is not None
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
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val) if has_val else None,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        class_weight=class_weight, callbacks=callbacks, verbose=1)
    return history