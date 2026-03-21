import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def generate_training_data(n_samples=2000, random_state=42):
    rng = np.random.RandomState(random_state)

    wind  = rng.uniform(0, 20, n_samples)
    load  = rng.uniform(5, 15, n_samples)
    soc   = rng.uniform(0, 100, n_samples)
    price = rng.choice([1.0, 3.0], n_samples)

    labels = []
    for w, l, s, p in zip(wind, load, soc, price):
        net = w - l
        if net > 1.0 and s < 90.0 and p == 1.0:
            labels.append("charge")
        elif net < 0.0 and s > 20.0 and p == 3.0:
            labels.append("discharge")
        elif net < 0.0 and s > 30.0 and p == 1.0:
            labels.append("discharge")
        else:
            labels.append("idle")

    X = np.column_stack([wind, load, soc, price])
    y = np.array(labels)
    return X, y


def load_model():
    X, y = generate_training_data(n_samples=2000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
    )
    clf.fit(X_train_s, y_train)

    acc = clf.score(X_test_s, y_test)
    print(f"[ML] RandomForest accuracy on hold-out set: {acc:.2%}")

    return clf, scaler


def predict(model_tuple, wind, load, battery, price=1.0):
    clf, scaler = model_tuple
    X = np.array([[wind, load, battery, price]])
    return clf.predict(scaler.transform(X))[0]