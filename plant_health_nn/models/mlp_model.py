"""MLP модель для класифікації стану здоров'я рослини на базі біосенсорних даних."""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_mlp(input_dim, num_classes=4, hidden_layers=None, dropout=0.3, learning_rate=0.001):
    """
    Будує многошарову персептронну мережу (MLP) для класифікації рослин.
    
    Архітектура: Dense(128) → BN → Drop → Dense(64) → BN → Drop → Dense(32) → BN → Drop → Dense(4-softmax)
    
    Args:
        input_dim (int): Кількість вхідних біосенсорних ознак (8)
        num_classes (int): Кількість класів для класифікації (за замовчуванням 4)
        hidden_layers (list): Розміри прихованих шарів, за замовчуванням [128, 64, 32]
        dropout (float): Коефіцієнт dropout для регуляризації (за замовчуванням 0.3)
        learning_rate (float): Швидкість навчання для Adam оптимізатора (за замовчуванням 0.001)
    
    Returns:
        keras.Model: Скомпільована модель з Adam оптимізатором та sparse categorical crossentropy loss
    
    Особливості:
        - L2 регуляризація (λ=1e-4) на всіх Dense шарах
        - BatchNormalization для стабілізації навчання
        - Dropout для запобігання overfitting
    """
    if hidden_layers is None:
        hidden_layers = [128, 64, 32]
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for units in hidden_layers:
        x = layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out, name="MLP_PlantHealth")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model