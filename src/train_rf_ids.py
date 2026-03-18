import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer
)
from sklearn.model_selection import RandomizedSearchCV


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


def rf_stage_decision(prob_attack: float, high_attack: float = 0.85, low_attack: float = 0.15):
    """
    Confidence-based routing logic for hybrid IDS stage 1.
    """
    if prob_attack >= high_attack:
        return "attack", "rf_only"
    elif prob_attack <= low_attack:
        return "normal", "rf_only"
    else:
        return "uncertain", "send_to_dnn"


def main():
    # 1) Load train/test
    train_path = "data/KDDTrain+.TXT"
    test_path = "data/KDDTest+.TXT"

    df_train = load_dataset_txt(train_path, 41)
    df_test = load_dataset_txt(test_path, 41)

    # 2) Basic sanity checks
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

    # 5) Preprocessing
    categorical_cols = ["f1", "f2", "f3"]
    numeric_cols = [f"f{i}" for i in range(41) if f"f{i}" not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    # 6) Base model
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    # 7) Pipeline
    model = Pipeline([
        ("prep", preprocessor),
        ("rf", rf)
    ])

    # 8) Hyperparameter tuning
    print("Running hyperparameter tuning...")

    attack_f1 = make_scorer(f1_score, pos_label=1)
    attack_recall = make_scorer(recall_score, pos_label=1)

    param_dist = {
    "rf__n_estimators": [200, 300, 500, 800, 1000],
    "rf__max_depth": [None, 10, 20, 30, 40, 60, 80],
    "rf__min_samples_split": [2, 5, 10, 20, 30],
    "rf__min_samples_leaf": [1, 2, 4, 8, 12],
    "rf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "rf__class_weight": [
        None,
        "balanced",
        {0: 1, 1: 2},
        {0: 1, 1: 3},
        {0: 1, 1: 4}
    ],
    "rf__bootstrap": [True, False]
}

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=40,
        scoring=attack_recall,
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(search.best_params_)

    best_model = search.best_estimator_

    # 9) Predict probabilities
    print("\nEvaluating tuned model on test set...")
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Baseline at default threshold 0.50
    print("\n--- Baseline Threshold = 0.50 ---")
    y_pred_default = (y_proba >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred_default)
    prec = precision_score(y_test, y_pred_default)
    rec = recall_score(y_test, y_pred_default)
    f1 = f1_score(y_test, y_pred_default)
    cm = confusion_matrix(y_test, y_pred_default)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("Confusion Matrix [ [TN FP], [FN TP] ]:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred_default, target_names=["normal(0)", "attack(1)"]))

    # Threshold experiments
    thresholds = [0.45, 0.40, 0.35, 0.30]
    threshold_results = []

    print("\n--- Threshold Tuning Results ---")
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred_thresh)
        prec = precision_score(y_test, y_pred_thresh)
        rec = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        cm = confusion_matrix(y_test, y_pred_thresh)

        threshold_results.append({
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        print(f"\nThreshold = {threshold}")
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1 Score:", f1)
        print("Confusion Matrix [ [TN FP], [FN TP] ]:\n", cm)

    # Pick best threshold by F1
    results_df = pd.DataFrame(threshold_results)
    best_row = results_df.loc[results_df["f1"].idxmax()]
    best_threshold = best_row["threshold"]

    print("\nBest threshold based on F1:")
    print(best_row)

    # Final predictions using best threshold
    y_pred = (y_proba >= best_threshold).astype(int)

    print(f"\n--- Final Selected Threshold = {best_threshold} ---")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["normal(0)", "attack(1)"]))

    results_df.to_csv("rf_threshold_tuning_results.csv", index=False)
    print("\nSaved threshold tuning results to: rf_threshold_tuning_results.csv")

    # 10) Show routing decisions for sample predictions
    print("\nSample RF stage-1 routing decisions:")
    for i in range(10):
        decision, route = rf_stage_decision(y_proba[i])
        print(
            f"Sample {i}: prob_attack={y_proba[i]:.4f}, "
            f"rf_prediction={y_pred[i]}, decision={decision}, route={route}"
        )

    # 11) Feature importance
    rf_model = best_model.named_steps["rf"]
    ohe = best_model.named_steps["prep"].named_transformers_["cat"]

    encoded_cat_names = ohe.get_feature_names_out(categorical_cols)
    all_feature_names = list(encoded_cat_names) + numeric_cols

    feature_importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": rf_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nTop 15 Feature Importances:")
    print(feature_importance_df.head(15))

    feature_importance_df.to_csv("rf_feature_importance.csv", index=False)
    print("\nSaved feature importance to: rf_feature_importance.csv")

    # 12) Save tuned model
    joblib.dump(best_model, "rf_ids_pipeline_tuned.joblib")
    print("\nSaved tuned pipeline to: rf_ids_pipeline_tuned.joblib")


if __name__ == "__main__":
    main()