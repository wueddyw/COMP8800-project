import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
import random

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Loading Data

def load_dataset_txt(path: str, features_count: int) -> pd.DataFrame:
    cols = [f"f{i}" for i in range(features_count)] + ["label", "difficulty"]
    return pd.read_csv(path, names=cols)


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    df["label_binary"] = (df["label"] != "normal").astype(int)
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=["label", "difficulty", "label_binary"])
    y = df["label_binary"]
    return X, y

# DNN Model
def build_dnn(input_dim: int):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Main pipeline
def main():
    train_path = "data/NSL-KDD/KDDTrain+.txt"
    test_path = "data/NSL-KDD/KDDTest+.txt"

    # Load dataset
    df_train = add_binary_label(load_dataset_txt(train_path, 41))
    df_test = add_binary_label(load_dataset_txt(test_path, 41))

    X_train, y_train = split_xy(df_train)
    X_test, y_test = split_xy(df_test)

    # Load tuned RF model
    rf_model = joblib.load("rf_ids_pipeline_tuned.joblib")

    # Reuse fitted preprocessing from RF pipeline
    preprocessor = rf_model.named_steps["prep"]

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse matrix to dense if needed
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # Normalize for DNN
    scaler = StandardScaler()
    X_train_dnn = scaler.fit_transform(X_train_processed)
    X_test_dnn = scaler.transform(X_test_processed)

    # Build and train DNN on full dataset
    print("\nTraining DNN on full dataset...")
    tf.keras.backend.clear_session()
    dnn = build_dnn(X_train_dnn.shape[1])

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)

    attack_multiplier = 1.5

    class_weight = {
        0: weights[0],
        1: weights[1] * attack_multiplier
    }

    print("Class weights:", class_weight)

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    dnn.fit(
        X_train_dnn,
        y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=256,
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=1
    )

    # RF probabilities
    test_proba = rf_model.predict_proba(X_test)[:, 1]

    # DNN only evaluation
    dnn_preds = (dnn.predict(X_test_dnn, verbose=0) >= 0.15).astype(int).flatten()

    print("\n--- DNN ONLY PERFORMANCE ---")
    print("Accuracy:", accuracy_score(y_test, dnn_preds))
    print("Precision:", precision_score(y_test, dnn_preds))
    print("Recall:", recall_score(y_test, dnn_preds))
    print("F1 Score:", f1_score(y_test, dnn_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, dnn_preds))

    # Hybrid prediction
    final_preds = []
    dnn_used_count = 0

    for i in range(len(X_test)):
        prob_attack = test_proba[i]

        # RF confident -> use RF (ONLY trust strong attacks)
        if prob_attack >= 0.75:
            final_preds.append(1)

        # Borderline RF attack -> auto classify as attack 
        elif prob_attack >= 0.43:
            final_preds.append(1)

        # EVERYTHING ELSE -> use DNN (including "normal")
        else:
            dnn_prob = dnn.predict(X_test_dnn[i:i+1], verbose=0)[0][0]
            dnn_pred = int(dnn_prob >= 0.15)

            final_preds.append(dnn_pred)
            dnn_used_count += 1

    final_preds = np.array(final_preds)

    # Results
    print("\n--- Hybrid IDS Results ---")
    print("DNN used on test samples:", dnn_used_count)
    print("RF-only decisions:", len(X_test) - dnn_used_count)

    acc = accuracy_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds)
    rec = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    cm = confusion_matrix(y_test, final_preds)

    print("\nAccuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)

    print("\nClassification Report:\n", classification_report(
        y_test,
        final_preds,
        target_names=["normal(0)", "attack(1)"]
    ))

    # Save model
    dnn.save("hybrid_dnn_model.keras")
    print("\nSaved DNN model to: hybrid_dnn_model.keras")


if __name__ == "__main__":
    main()