import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ─── 1. LOAD & CLEAN DATA ────────────────────────────────────────────────────

def load_and_clean_data(path="data/real_estate_listings_india_2025.csv"):
    df = pd.read_csv(path)

    def convert_size(size):
        try:
            size = str(size).replace(" sqft", "").replace("sq.ft.", "").strip()
            size = size.replace(",", "")
            if "-" in size:
                parts = size.split("-")
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p.strip()))
                    except:
                        pass
                return np.mean(nums) if nums else np.nan
            return float(size)
        except:
            return np.nan

    df["size_avg"] = df["size"].apply(convert_size)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price", "size_avg"])
    df = df[df["price"] > 100000]
    df = df[df["size_avg"] > 0]
    df.drop(columns=["url", "date", "size"], inplace=True, errors="ignore")
    df.dropna(inplace=True)

    return df


# ─── 2. FEATURE ENGINEERING ──────────────────────────────────────────────────

def engineer_features(df):
    """Add derived features and create price segment target."""

    # Price segment (target)
    def price_segment(price):
        if price < 5_000_000:
            return "Budget"
        elif price < 10_000_000:
            return "MidRange"
        elif price < 20_000_000:
            return "Premium"
        else:
            return "Luxury"

    df["PriceSegment"] = df["price"].apply(price_segment)

    # Price per sqft
    df["price_per_sqft"] = df["price"] / df["size_avg"]

    # BHK category
    df["bhk_category"] = df["beds"].apply(
        lambda x: "Studio/1BHK" if x <= 1 else ("2BHK" if x == 2 else ("3BHK" if x == 3 else "4+BHK"))
    )

    return df


# ─── 3. TRAIN/TEST SPLIT & PREPROCESSING ─────────────────────────────────────

def build_pipeline(model, categorical_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="passthrough"
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])


def evaluate_model(pipeline, X, y, kf):
    acc_scores, f1_scores = [], []
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc_scores.append(accuracy_score(y_test, preds))
        f1_scores.append(f1_score(y_test, preds, average="weighted"))

    return {
        "Accuracy": round(np.mean(acc_scores), 4),
        "F1_Score": round(np.mean(f1_scores), 4),
        "Acc_Std": round(np.std(acc_scores), 4)
    }


# ─── 4. MAIN PIPELINE ────────────────────────────────────────────────────────

def run_supervised_pipeline(data_path="data/real_estate_listings_india_2025.csv"):
    print("=" * 60)
    print("  Real Estate Price Segment Prediction Pipeline")
    print("=" * 60)

    # Load & engineer
    print("\n[1/4] Loading and preprocessing data...")
    df = load_and_clean_data(data_path)
    df = engineer_features(df)
    print(f"      Dataset shape: {df.shape}")
    print(f"      Segment distribution:\n{df['PriceSegment'].value_counts().to_string()}")

    # Features
    feature_cols = ["beds", "baths", "size_avg", "city", "type", "neighborhood", "bhk_category"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["PriceSegment"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Label encoder for final model
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Define all 5 models
    models = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, max_depth=10, min_samples_split=4,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="mlogloss", verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            num_leaves=20, random_state=42, verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=150, depth=4, learning_rate=0.05,
            random_seed=42, verbose=0
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100, learning_rate=0.1,
            random_state=42
        )
    }

    # Build pipelines (CatBoost handles cats natively, others need OHE)
    pipelines = {}
    for name, model in models.items():
        if name == "CatBoost":
            # CatBoost can handle categoricals natively
            cat_idx = [X.columns.get_loc(c) for c in categorical_cols]
            pipelines[name] = ("catboost_native", model, categorical_cols, cat_idx)
        else:
            pipelines[name] = build_pipeline(model, categorical_cols)

    # Cross-validation
    print("\n[2/4] Running 5-Fold Cross-Validation on all models...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, pipe in pipelines.items():
        print(f"      Evaluating {name}...", end=" ")
        try:
            if name == "CatBoost":
                _, cb_model, cat_cols, _ = pipe
                acc_scores, f1_scores = [], []
                for train_idx, test_idx in kf.split(X):
                    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
                    y_train, y_test = y_enc[train_idx], y_enc[test_idx]
                    # Convert cats to string for CatBoost
                    for c in cat_cols:
                        X_train[c] = X_train[c].astype(str)
                        X_test[c] = X_test[c].astype(str)
                    cb_model.fit(X_train, y_train, cat_features=cat_cols, verbose=0)
                    preds = cb_model.predict(X_test)
                    acc_scores.append(accuracy_score(y_test, preds))
                    f1_scores.append(f1_score(y_test, preds, average="weighted"))
                results[name] = {
                    "Accuracy": round(np.mean(acc_scores), 4),
                    "F1_Score": round(np.mean(f1_scores), 4),
                    "Acc_Std": round(np.std(acc_scores), 4)
                }
            else:
                results[name] = evaluate_model(pipe, X, y, kf)
            print(f"Accuracy={results[name]['Accuracy']:.4f}, F1={results[name]['F1_Score']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[name] = {"Accuracy": 0, "F1_Score": 0, "Acc_Std": 0}

    # Model evaluation summary
    print("\n[3/4] Model Evaluation Results:")
    print("-" * 55)
    print(f"  {'Model':<15} {'Accuracy':>10} {'F1 Score':>10} {'Std Dev':>10}")
    print("-" * 55)
    for name, res in results.items():
        print(f"  {name:<15} {res['Accuracy']:>10.4f} {res['F1_Score']:>10.4f} {res['Acc_Std']:>10.4f}")
    print("-" * 55)

    # Best model selection (by F1 score)
    best_name = max(results, key=lambda k: results[k]["F1_Score"])
    print(f"\n  ✓ Best Model Selected: {best_name} (F1={results[best_name]['F1_Score']:.4f})")

    # Train final best model on full data
    print("\n[4/4] Training final model on full dataset...")
    if best_name == "CatBoost":
        _, final_model, cat_cols, _ = pipelines[best_name]
        X_final = X.copy()
        for c in cat_cols:
            X_final[c] = X_final[c].astype(str)
        final_model.fit(X_final, y_enc, cat_features=cat_cols, verbose=0)
        final_pipeline = ("catboost_final", final_model, cat_cols, le)
    else:
        final_pipeline = pipelines[best_name]
        final_pipeline.fit(X, y_enc)

    print(f"  ✓ Final model trained and ready for predictions!")

    # Save results
    output = {
        "best_model": best_name,
        "results": results,
        "label_classes": le.classes_.tolist(),
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols
    }

    import os
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model_results.json", "w") as f:
        json.dump(output, f, indent=2)

    return final_pipeline, le, feature_cols, categorical_cols, results, best_name


# ─── 5. PRICE SEGMENT PREDICTION ENGINE ──────────────────────────────────────

def predict_price_segment(user_input: dict, final_pipeline, le, feature_cols, categorical_cols):
    """
    Price Segment Prediction Engine
    Input: user property details dict
    Output: predicted price segment + probabilities
    """
    # Build input dataframe
    input_df = pd.DataFrame([user_input])

    # Add derived features
    if "beds" in input_df.columns:
        beds = input_df["beds"].iloc[0]
        input_df["bhk_category"] = (
            "Studio/1BHK" if beds <= 1 else
            "2BHK" if beds == 2 else
            "3BHK" if beds == 3 else "4+BHK"
        )

    # Ensure all needed columns present
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = "Unknown" if col in categorical_cols else 0

    input_df = input_df[feature_cols]

    # Predict
    if isinstance(final_pipeline, tuple) and final_pipeline[0] == "catboost_final":
        _, model, cat_cols, _ = final_pipeline
        for c in cat_cols:
            input_df[c] = input_df[c].astype(str)
        pred_enc = model.predict(input_df)[0]
        try:
            probas = model.predict_proba(input_df)[0]
        except:
            probas = None
    else:
        pred_enc = final_pipeline.predict(input_df)[0]
        try:
            probas = final_pipeline.predict_proba(input_df)[0]
        except:
            probas = None

    predicted_segment = le.inverse_transform([int(pred_enc)])[0]

    segment_info = {
        "Budget":   "< ₹50 Lakh",
        "MidRange": "₹50L – ₹1 Cr",
        "Premium":  "₹1 Cr – ₹2 Cr",
        "Luxury":   "> ₹2 Cr"
    }

    result = {
        "predicted_segment": predicted_segment,
        "price_range": segment_info.get(predicted_segment, "N/A"),
        "confidence": {}
    }

    if probas is not None:
        for i, cls in enumerate(le.classes_):
            result["confidence"][cls] = round(float(probas[i]) * 100, 1)

    return result


# ─── 6. DEMO RUN ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run full pipeline
    final_pipeline, le, feature_cols, categorical_cols, results, best_name = run_supervised_pipeline()

    print("\n" + "=" * 60)
    print("  PRICE SEGMENT PREDICTION ENGINE — Demo")
    print("=" * 60)

    test_cases = [
        {
            "beds": 2, "baths": 2, "size_avg": 1200,
            "city": "Hyderabad", "type": "2 BHK Flat", "neighborhood": "Madhapur"
        },
        {
            "beds": 4, "baths": 4, "size_avg": 4200,
            "city": "Gurgaon", "type": "4 BHK Flat", "neighborhood": "Sector 65"
        },
        {
            "beds": 1, "baths": 1, "size_avg": 600,
            "city": "Mumbai", "type": "1 BHK Flat", "neighborhood": "Kandivali East"
        },
        {
            "beds": 3, "baths": 3, "size_avg": 1800,
            "city": "Bangalore", "type": "3 BHK Apartment", "neighborhood": "Whitefield"
        },
    ]

    for i, case in enumerate(test_cases, 1):
        result = predict_price_segment(case, final_pipeline, le, feature_cols, categorical_cols)
        print(f"\nTest Case {i}: {case['beds']}BHK in {case['city']} ({case['neighborhood']})")
        print(f"  → Predicted Segment : {result['predicted_segment']}")
        print(f"  → Price Range       : {result['price_range']}")
        if result["confidence"]:
            conf_str = " | ".join([f"{k}: {v}%" for k, v in sorted(result["confidence"].items(), key=lambda x: -x[1])])
            print(f"  → Confidence        : {conf_str}")

    print("\n✓ All done! Results saved to outputs/model_results.json")