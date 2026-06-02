from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    if not val_size:  # if val_size is 0, 0.0, None or False
        return X_train, None, X_test, y_train, None, y_test
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=random_state, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test
