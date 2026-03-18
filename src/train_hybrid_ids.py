import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


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


def build_dnn(input_dim: int) -> Sequential:
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    train_path = "data/NSL-KDD/KDDTrain+.txt"
    test_path = "data/NSL-KDD/KDDTest+.txt"

    df_train = add_binary_label(load_dataset_txt(train_path, 41))
    df_test = add_binary_label(load_dataset_txt(test_path, 41))

    X_train, y_train = split_xy(df_train)
    X_test, y_test = split_xy(df_test)

    # Load tuned RF pipeline
    rf_model = joblib.load("rf_ids_pipeline_tuned.joblib")

    # Reuse fitted preprocessing from RF pipeline
    preprocessor = rf_model.named_steps["prep"]

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # RF probabilities
    train_proba = rf_model.predict_proba(X_train)[:, 1]
    test_proba = rf_model.predict_proba(X_test)[:, 1]

    # Confidence band for hybrid routing
    low_threshold = 0.15
    high_threshold = 0.85

    uncertain_train_mask = (train_proba > low_threshold) & (train_proba < high_threshold)

    X_train_dnn = X_train_processed[uncertain_train_mask]
    y_train_dnn = y_train[uncertain_train_mask]

    print("Total train samples:", len(X_train))
    print("Uncertain samples for DNN training:", X_train_dnn.shape[0])

    # Convert sparse matrix to dense if needed
    if hasattr(X_train_dnn, "toarray"):
        X_train_dnn = X_train_dnn.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # Build and train DNN
    dnn = build_dnn(X_train_dnn.shape[1])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    dnn.fit(
        X_train_dnn,
        y_train_dnn,
        validation_split=0.2,
        epochs=10,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1
    )

    # Hybrid prediction
    final_preds = []
    dnn_used_count = 0

    for i in range(len(X_test)):
        prob_attack = test_proba[i]

        if prob_attack >= high_threshold:
            final_preds.append(1)
        elif prob_attack <= low_threshold:
            final_preds.append(0)
        else:
            dnn_prob = dnn.predict(X_test_processed[i:i+1], verbose=0)[0][0]
            dnn_pred = int(dnn_prob >= 0.5)
            final_preds.append(dnn_pred)
            dnn_used_count += 1

    final_preds = np.array(final_preds)

    print("\n--- Hybrid IDS Results ---")
    print("DNN used on test samples:", dnn_used_count)
    print("Accuracy:", accuracy_score(y_test, final_preds))
    print("Precision:", precision_score(y_test, final_preds))
    print("Recall:", recall_score(y_test, final_preds))
    print("F1 Score:", f1_score(y_test, final_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))
    print("\nClassification Report:\n", classification_report(
        y_test,
        final_preds,
        target_names=["normal(0)", "attack(1)"]
    ))

    dnn.save("hybrid_dnn_model.keras")
    print("\nSaved DNN model to: hybrid_dnn_model.keras")


if __name__ == "__main__":
    main()