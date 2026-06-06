"""CNN модель для класифікації стану здоров'я рослини з вилученням часових ознак."""
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_dim, num_classes=4, learning_rate=0.001):
    """
    Будує одновимірну згорткову нейронну мережу (CNN) для класифікації рослин.
    
    Архітектура: Conv1D(64,3) → BN → Conv1D(128,3) → BN → GlobalAvgPool → Dense(64) → Drop → Dense(4-softmax)
    
    Args:
        input_dim (int): Кількість вхідних ознак (8), буде перетворено на (8, 1) для Conv1D
        num_classes (int): Кількість класів для класифікації (за замовчуванням 4)
        learning_rate (float): Швидкість навчання для Adam оптимізатора (за замовчуванням 0.001)
    
    Returns:
        keras.Model: Скомпільована CNN модель для 1D сигналів
    
    Особливості:
        - Conv1D шари для вилучення локальних закономірностей в послідовності ознак
        - GlobalAveragePooling для агрегації активацій
        - BatchNormalization для стабілізації
        - Dropout для регуляризації
    """
    inp = layers.Input(shape=(input_dim, 1))
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out, name="CNN_PlantHealth")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
