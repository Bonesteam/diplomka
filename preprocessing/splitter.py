from sklearn.model_selection import train_test_split


def split_train_val(X, y, val_size=0.2, random_state=42):
    """Stratified train/validation split (наприклад, 80/20)."""
    return train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )


def split_train_val_test(X, y, train_size=0.64, val_size=0.16, test_size=0.2, random_state=42):
    """
    Stratified 3-way split: train / validation / test.

    Default 64/16/20 = 80% навчальний пул (train+val) + 20% ізольований test.
    З навчального пулу 80% → train, 20% → val (тобто 64% і 16% від усього датасету).

    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Sizes must sum to 1.0, got {total}")

    # First split: (train+val) pool vs isolated test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val within the pool
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_from_config(X, y, config):
    """Split using sizes from config.yaml data section."""
    data_cfg = config.get("data", {})
    return split_train_val_test(
        X, y,
        train_size=data_cfg.get("train_size", 0.64),
        val_size=data_cfg.get("val_size", 0.16),
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
    )


def format_split_summary(y_train, y_val, y_test, config=None):
    """Human-readable split description for logs and thesis."""
    total = len(y_train) + len(y_val) + len(y_test)
    strategy = (config or {}).get("data", {}).get(
        "split_strategy",
        "80/20 (test isolated); val=20% of train pool → 64/16/20",
    )
    return {
        "strategy": strategy,
        "train": len(y_train),
        "val": len(y_val),
        "test": len(y_test),
        "train_pct": len(y_train) / total * 100,
        "val_pct": len(y_val) / total * 100,
        "test_pct": len(y_test) / total * 100,
    }


def split_data(X, y, test_size=0.2, val_size=0.16, random_state=42):
    """LEGACY: Old 3-way split function. Use split_train_val_test instead."""
    train_size = 1.0 - test_size - val_size
    return split_train_val_test(
        X, y,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )
