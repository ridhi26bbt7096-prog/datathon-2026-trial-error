"""
SlotSwap wait-time ML model
---------------------------
Train and evaluate a Random Forest regression model on encounters.csv.

Usage:
    python model.py --data data/encounters.csv
    python model.py --data encounters.csv --output slotswap_wait_model.joblib

Outputs:
    - slotswap_wait_model.joblib: trained scikit-learn pipeline + feature metadata
    - model_metrics.json: held-out test metrics
    - test_predictions.csv: actual vs predicted wait times

Important: this is a hackathon demonstration using the supplied dataset.
It is not a clinical decision tool and must not be used to promise real wait times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    for col in df.columns:
        low = str(col).strip().lower()
        if any(candidate.lower() in low for candidate in candidates):
            return col
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SlotSwap wait-time regression model.")
    parser.add_argument("--data", default="data/encounters.csv", help="Path to encounters.csv")
    parser.add_argument("--output", default="slotswap_wait_model.joblib", help="Output model artifact")
    parser.add_argument("--metrics", default="model_metrics.json", help="Output metrics JSON")
    parser.add_argument("--predictions", default="test_predictions.csv", help="Output held-out predictions CSV")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Could not find {data_path}. Put encounters.csv in the data folder "
            "or pass --data PATH_TO_CSV."
        )

    df = pd.read_csv(data_path)
    wait_col = find_col(df, ["wait_time_minutes", "waiting_time", "wait_minutes", "wait_time", "wait"])
    if wait_col is None:
        raise SystemExit(
            "Couldn't identify the wait-time column. Columns found: "
            + ", ".join(map(str, df.columns))
        )

    dept_col = find_col(df, ["department", "department_name", "dept"])
    triage_col = find_col(df, ["triage_level", "triage", "acuity"])
    visit_type_col = find_col(df, ["visit_type", "type"])
    arrival_col = find_col(df, ["arrival_time", "arrival_datetime", "check_in_time"])

    features = pd.DataFrame(index=df.index)
    categorical_cols: list[str] = []
    for col in [dept_col, triage_col, visit_type_col]:
        if col is not None and col not in categorical_cols:
            features[col] = df[col].fillna("Unknown").astype(str)
            categorical_cols.append(col)

    if arrival_col is not None:
        arrival = pd.to_datetime(df[arrival_col], errors="coerce")
        if arrival.notna().sum() > 0:
            features["arrival_hour"] = arrival.dt.hour
            features["arrival_day_of_week"] = arrival.dt.dayofweek

    y = pd.to_numeric(df[wait_col], errors="coerce")
    valid = y.notna()
    features = features.loc[valid].copy()
    y = y.loc[valid].clip(lower=0)

    if features.shape[1] == 0:
        raise SystemExit(
            "No usable feature columns found. Expected at least a department, triage, "
            "visit type, or parseable arrival timestamp column."
        )
    if len(features) < 30 or y.nunique() < 2:
        raise SystemExit("Need at least 30 usable rows and more than one wait-time value.")

    categorical_cols = list(features.select_dtypes(include=["object"]).columns)
    numeric_cols = [col for col in features.columns if col not in categorical_cols]

    transformers = []
    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", cat_pipeline, categorical_cols))
    if numeric_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("numeric", num_pipeline, numeric_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=160,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    predictions = np.maximum(0, pipeline.predict(X_test))

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "feature_columns": list(features.columns),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "target_column": str(wait_col),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Synthetic-data hackathon demo; not for clinical decisions.",
    }, output_path)

    metrics = {
        "model": "RandomForestRegressor",
        "target": str(wait_col),
        "rows_used": int(len(features)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": list(features.columns),
        "mae_minutes": mae,
        "rmse_minutes": rmse,
        "r2": r2,
        "random_state": 42,
        "test_size": 0.2,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "caveat": "Synthetic dataset; estimates are not clinical advice or live wait-time guarantees.",
    }
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pd.DataFrame({
        "actual_wait_minutes": y_test.to_numpy(),
        "predicted_wait_minutes": predictions,
        "absolute_error_minutes": np.abs(y_test.to_numpy() - predictions),
    }).to_csv(args.predictions, index=False)

    print("\nSLOTSWAP WAIT-TIME MODEL — HELD-OUT EVALUATION")
    print("=" * 48)
    print(f"Target column : {wait_col}")
    print(f"Rows          : {len(features):,}")
    print(f"Train / test  : {len(X_train):,} / {len(X_test):,}")
    print(f"Features      : {', '.join(map(str, features.columns))}")
    print(f"MAE           : {mae:.2f} minutes")
    print(f"RMSE          : {rmse:.2f} minutes")
    print(f"R²            : {r2:.3f}")
    print(f"\nSaved model     : {output_path}")
    print(f"Saved metrics   : {args.metrics}")
    print(f"Saved predictions: {args.predictions}")
    print("\nWarning: demonstration model trained on supplied data; not for clinical decisions.")


if __name__ == "__main__":
    main()
