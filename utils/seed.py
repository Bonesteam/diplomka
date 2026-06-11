"""Фіксація глобального seed для повної відтворюваності результатів."""
import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Фіксує seed для Python, NumPy та TensorFlow/Keras.
    Викликати до будь-якого імпорту tensorflow і до побудови моделі.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow імпортується тут щоб не уповільнювати старт програми
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
