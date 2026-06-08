"""Модуль для передбачення та завантаження моделей."""
import numpy as np
import tensorflow as tf


def get_probabilities(model, X_scaled):
    """Сирі ймовірності моделі."""
    probs = model.predict(X_scaled, verbose=0)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    return np.asarray(probs, dtype=np.float64)


def load_model(path):
    """Завантажує збережену Keras модель."""
    return tf.keras.models.load_model(path)
