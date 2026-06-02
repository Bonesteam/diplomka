import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_dim, num_classes=4, learning_rate=0.001):
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
