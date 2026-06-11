import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from models.mlp_model import build_mlp

def cross_validate_mlp(X, y, config, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        print(f"  Фолд {fold+1}/{n_splits}")
        model = build_mlp(X.shape[1], hidden_layers=config["model"]["hidden_layers"],
                          dropout=config["model"]["dropout"],
                          learning_rate=config["training"]["learning_rate"])
        cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                               restore_best_weights=True, verbose=0)
        model.fit(X[tr], y[tr], validation_data=(X[val], y[val]),
                  epochs=config["training"]["epochs"],
                  batch_size=config["training"]["batch_size"], callbacks=[cb], verbose=0)
        y_pred = np.argmax(model.predict(X[val], verbose=0), axis=1)
        accs.append(accuracy_score(y[val], y_pred))
        f1s.append(f1_score(y[val], y_pred, average="weighted"))
        print(f"    acc={accs[-1]:.4f}  f1={f1s[-1]:.4f}")
    print(f"  Середнє: acc={np.mean(accs):.4f}±{np.std(accs):.4f}  f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
    return {"accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
