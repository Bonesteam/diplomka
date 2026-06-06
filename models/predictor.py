"""Модуль для передбачення та завантаження моделей."""
import numpy as np
import tensorflow as tf


def predict(model, X):
    """
    Робить передбачення на вхідних даних.

    Returns:
        tuple: (передбачені класи, ймовірності для всіх класів)
    """
    probs = model.predict(X, verbose=0)
    return np.argmax(probs, axis=1), probs


def get_probabilities(model, X_scaled):
    """Сирі ймовірності моделі (без штучного згладжування)."""
    probs = model.predict(X_scaled, verbose=0)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    return np.asarray(probs, dtype=np.float64)


def load_model(path):
    """Завантажує збережену Keras модель з файлу."""
    return tf.keras.models.load_model(path)
