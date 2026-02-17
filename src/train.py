import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib


def load_dataset_txt(path: str, features_count: int) -> pd.DataFrame:
    cols = [f"f{i}" for i in range(features_count)] + ["label", "difficulty"]
    data_frame = pd.read_csv(path, names=cols)
    return data_frame


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    # Keep original label; add a binary target for prototype
    df["label_binary"] = (df["label"] != "normal").astype(int)
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=["label", "difficulty", "label_binary"])
    y = df["label_binary"]
    return X, y


def main():
    # 1) Load train/test
    train_path = "data/KDDTrain+.TXT"
    test_path = "data/KDDTest+.TXT"

    df_train = load_dataset_txt(train_path, 41)
    df_test = load_dataset_txt(test_path, 41)

    # 2) Basic sanity checks (optional but helpful)
    print("Train head:\n", df_train.head(), "\n")
    print("Train original label counts (top 10):\n", df_train["label"].value_counts().head(10), "\n")

    # 3) Binary labels
    df_train = add_binary_label(df_train)
    df_test = add_binary_label(df_test)

    print("Binary label counts (train):\n", df_train["label_binary"].value_counts(), "\n")
    print("Binary label counts (test):\n", df_test["label_binary"].value_counts(), "\n")

    # 4) Separate X/y
    X_train, y_train = split_xy(df_train)
    X_test, y_test = split_xy(df_test)

    print("X_train shape:", X_train.shape, " y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape, " y_test shape:", y_test.shape, "\n")

    # 5) Preprocessing (one-hot encode categorical columns)
    # In NSL-KDD TXT, f1/f2/f3 are protocol_type/service/flag
    categorical_cols = ["f1", "f2", "f3"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    # 6) Model
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # 7) Pipeline
    model = Pipeline([
        ("prep", preprocessor),
        ("rf", rf)
    ])

    # 8) Train
    print("Training Random Forest...")
    model.fit(X_train, y_train)

    # 9) Predict + Evaluate
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nAccuracy:", acc)
    print("\nConfusion Matrix [ [TN FP], [FN TP] ]:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["normal(0)", "attack(1)"]))

    # 10) Optional: save for API demo
    joblib.dump(model, "rf_ids_pipeline.joblib")
    print("\nSaved trained pipeline to: rf_ids_pipeline.joblib")


if __name__ == "__main__":
    main()
