import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "HandDataset.csv")  # ✅ merged multi-user file
MODEL_FILE = os.path.join(os.path.dirname(__file__), "general_mlp.pkl")

def train_general_mlp():
    if not os.path.exists(DATA_FILE):
        print(f"Dataset not found at {DATA_FILE}. Put your merged multi-user dataset here.")
        return

    df = pd.read_csv(DATA_FILE)
    if df.empty:
        print("Dataset is empty. Cannot train model.")
        return

    from sklearn.preprocessing import LabelEncoder
    
    X = df.drop('label', axis=1).values
    
    # ⚠️ FIX: MLPClassifier's internal validation score function crashes if y is strings.
    # We must explicitly encode the string labels into integers.
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)
    
    print("\nLabel Mapping encoded dynamically:")
    for idx, class_name in enumerate(le.classes_):
        print(f"  {idx} -> {class_name}")

    # ✅ stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # ✅ Pipeline = scaler + MLP (export one file)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_FILE)
    print(f"\n✅ Saved gatekeeper model to: {MODEL_FILE}")
    print("Use model.predict_proba() during calibration to validate + gate recording.")

if __name__ == "__main__":
    train_general_mlp()