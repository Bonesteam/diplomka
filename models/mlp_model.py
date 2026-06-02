import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_mlp(input_dim, num_classes=4, hidden_layers=None, dropout=0.3, learning_rate=0.001):
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