"""Модуль для передбачення та завантаження моделей."""
import numpy as np, tensorflow as tf

def predict(model, X):
    """
    Робить передбачення на вхідних даних.
    
    Args:
        model: Keras модель
        X: Вхідні дані форми (n_samples, n_features)
    
    Returns:
        tuple: (передбачені класи, ймовірності для всіх класів)
    """
    probs = model.predict(X, verbose=0)
    return np.argmax(probs, axis=1), probs

def load_model(path):
    """
    Завантажує збережену Keras модель з файлу.
    
    Args:
        path (str): Шлях до файлу моделі (.keras формат)
    
    Returns:
        keras.Model: Завантажена модель
    """
    return tf.keras.models.load_model(path)
