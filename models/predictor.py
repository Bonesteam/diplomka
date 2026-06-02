import numpy as np, tensorflow as tf

def predict(model, X):
    probs = model.predict(X, verbose=0)
    return np.argmax(probs, axis=1), probs

def load_model(path):
    return tf.keras.models.load_model(path)
