from sklearn.model_selection import train_test_split


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Один датасет → 80% train / 20% test (sklearn train_test_split).
    Val для графіків — автоматично в Keras через validation_split під час fit.
    Returns: X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def split_from_config(X, y, config):
    """Split using sizes from config.yaml data section."""
    data_cfg = config.get("data", {})
    return split_train_test(
        X, y,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
    )


def format_split_summary(y_train, y_test, config=None):
    """Human-readable split description for logs."""
    total = len(y_train) + len(y_test)
    val_split = (config or {}).get("training", {}).get("validation_split", 0.2)
    strategy = (config or {}).get("data", {}).get(
        "split_strategy",
        get_split_description(val_split),
    )
    return {
        "strategy": strategy,
        "train": len(y_train),
        "test": len(y_test),
        "val_split": val_split,
        "train_pct": len(y_train) / total * 100,
        "test_pct": len(y_test) / total * 100,
    }


def get_split_description(val_split=0.2):
    """Єдиний текст про розбиття для UI, логів і config."""
    return (
        f"80/20 train/test (train_test_split); "
        f"val-криві через Keras validation_split={val_split}"
    )
