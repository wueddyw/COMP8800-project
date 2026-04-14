import os
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


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


# Main
def main():
    train_path = "data/NSL-KDD/KDDTrain+.txt"
    test_path = "data/NSL-KDD/KDDTest+.txt"

    rf_model_path = "rf_ids_pipeline_tuned.joblib"
    dnn_model_path = "hybrid_dnn_model.keras"

    output_dir = "explanations"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df_train = add_binary_label(load_dataset_txt(train_path, 41))
    df_test = add_binary_label(load_dataset_txt(test_path, 41))

    X_train, y_train = split_xy(df_train)
    X_test, y_test = split_xy(df_test)

    # Load trained models
    rf_pipeline = joblib.load(rf_model_path)
    dnn_model = load_model(dnn_model_path)

    # Reuse preprocessing from RF pipeline
    preprocessor = rf_pipeline.named_steps["prep"]
    rf_estimator = rf_pipeline.named_steps["rf"]

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse to dense
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # Recreate scaler used for DNN-style inputs
    scaler = StandardScaler()
    X_train_dnn = scaler.fit_transform(X_train_processed)
    X_test_dnn = scaler.transform(X_test_processed)

    # Expanded feature names after one-hot encoding
    categorical_cols = ["f1", "f2", "f3"]
    numeric_cols = [f"f{i}" for i in range(41) if f"f{i}" not in categorical_cols]

    ohe = preprocessor.named_transformers_["cat"]
    encoded_cat_names = list(ohe.get_feature_names_out(categorical_cols))
    feature_names = encoded_cat_names + numeric_cols

    # Pick one example to explain
    attack_indices = np.where(y_test.values == 1)[0]
    if len(attack_indices) == 0:
        raise ValueError("No attack samples found in test set.")

    sample_index = int(attack_indices[0])

    print(f"Explaining test sample index: {sample_index}")
    print(f"Ground truth label: {y_test.iloc[sample_index]}")

    # SHAP for RF stage
    print("\nRunning SHAP for Random Forest...")

    background_size = min(500, X_train_processed.shape[0])
    shap_background = X_train_processed[:background_size]

    shap_explainer = shap.TreeExplainer(rf_estimator, data=shap_background)

    # Explain one sample
    shap_single = shap_explainer(X_test_processed[sample_index:sample_index + 1])

    plt.figure()
    shap.plots.waterfall(shap_single[0, :, 1], max_display=15, show=False)
    plt.savefig(os.path.join(output_dir, "shap_waterfall_rf.png"), bbox_inches="tight")
    plt.close()

    # Explain a subset for summary plot
    summary_size = min(300, X_test_processed.shape[0])
    shap_summary = shap_explainer(X_test_processed[:summary_size])

    summary_values = shap_summary.values
    if summary_values.ndim == 3:
        summary_values = summary_values[:, :, 1]  # attack class

    plt.figure()
    shap.summary_plot(
        summary_values,
        X_test_processed[:summary_size],
        feature_names=feature_names,
        show=False
    )
    plt.savefig(os.path.join(output_dir, "shap_summary_rf.png"), bbox_inches="tight")
    plt.close()

    print("Saved SHAP plots:")
    print("-", os.path.join(output_dir, "shap_waterfall_rf.png"))
    print("-", os.path.join(output_dir, "shap_summary_rf.png"))

    # Hybrid prediction function for LIME
    # Input expected: already-preprocessed RF feature space
    def hybrid_predict_proba(processed_rows: np.ndarray) -> np.ndarray:
        results = []

        # Same thresholds as your final setup
        strong_rf_attack_threshold = 0.75
        borderline_rf_attack_threshold = 0.45
        dnn_attack_threshold = 0.15

        processed_rows = np.asarray(processed_rows)

        # Scale rows for DNN
        dnn_rows = scaler.transform(processed_rows)

        for i in range(processed_rows.shape[0]):
            rf_attack_prob = rf_estimator.predict_proba(processed_rows[i:i+1])[0, 1]

            if rf_attack_prob >= strong_rf_attack_threshold:
                attack_prob = 1.0
            elif rf_attack_prob >= borderline_rf_attack_threshold:
                attack_prob = 1.0
            else:
                dnn_attack_prob = float(dnn_model.predict(dnn_rows[i:i+1], verbose=0)[0][0])
                attack_prob = 1.0 if dnn_attack_prob >= dnn_attack_threshold else 0.0

            results.append([1.0 - attack_prob, attack_prob])

        return np.array(results)

    # LIME for final hybrid decision
    print("\nRunning LIME for Hybrid IDS...")

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_processed,
        feature_names=feature_names,
        class_names=["normal", "attack"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    lime_exp = lime_explainer.explain_instance(
        data_row=X_test_processed[sample_index],
        predict_fn=hybrid_predict_proba,
        num_features=15,
        top_labels=2
    )

    lime_html_path = os.path.join(output_dir, "lime_hybrid_explanation.html")
    lime_exp.save_to_file(lime_html_path)

    fig = lime_exp.as_pyplot_figure(label=1)
    fig.savefig(os.path.join(output_dir, "lime_hybrid_explanation.png"), bbox_inches="tight")
    plt.close(fig)

    print("Saved LIME outputs:")
    print("-", lime_html_path)
    print("-", os.path.join(output_dir, "lime_hybrid_explanation.png"))

    # Print console explanation
    rf_prob = rf_estimator.predict_proba(X_test_processed[sample_index:sample_index + 1])[0, 1]
    hybrid_prob = hybrid_predict_proba(X_test_processed[sample_index:sample_index + 1])[0, 1]

    print("\n--- Sample Prediction Summary ---")
    print("RF attack probability:", rf_prob)
    print("Hybrid final attack probability:", hybrid_prob)
    print("True label:", y_test.iloc[sample_index])

    print("\nTop LIME features for attack:")
    for feature, weight in lime_exp.as_list(label=1):
        print(f"{feature}: {weight:.4f}")


if __name__ == "__main__":
    main()