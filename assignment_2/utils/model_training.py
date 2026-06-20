import json
import os
import pickle
import pprint
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from utils.constants import (
    GOLD_LOANS_PATH, LABEL_DEF, MODEL_BANK_DIR, MODEL_TRAIN_DATE_STR,
    OOT_MONTHS, TEST_RATIO, TRAIN_RATIO, TRAIN_TEST_MONTHS, VAL_RATIO,
)

ID_COLS = ["Customer_ID", "loan_id", "snapshot_date"]
LABEL_COLS = ["label", "label_def"]
CATEGORICAL_COLS = ["Occupation", "Age_group"]
RANDOM_STATE = 611

# logistic_regression is the current (champion) production model, prioritised for
# explainability. xgboost runs alongside as a challenger to track the performance ceiling.
MODEL_ROLES = {
    "logistic_regression": "champion",
    "xgboost": "challenger",
}


def load_training_data(path=GOLD_LOANS_PATH):
    df = pd.read_parquet(path)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def get_feature_columns(df):
    exclude = set(ID_COLS + LABEL_COLS + CATEGORICAL_COLS)
    numeric_cols = [c for c in df.columns if c not in exclude]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    return numeric_cols, categorical_cols


def build_config(model_train_date_str=MODEL_TRAIN_DATE_STR, train_test_period_months=TRAIN_TEST_MONTHS,
                 oot_period_months=OOT_MONTHS, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    config = {
        "model_train_date_str": model_train_date_str,
        "train_test_period_months": train_test_period_months,
        "oot_period_months": oot_period_months,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    config["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["oot_end_date"] = config["model_train_date"] - timedelta(days=1)
    config["oot_start_date"] = config["model_train_date"] - relativedelta(months=oot_period_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_period_months)
    config["oot_month_ranges"] = [
        (config["oot_start_date"] + relativedelta(months=i),
         config["oot_start_date"] + relativedelta(months=i + 1) - timedelta(days=1))
        for i in range(oot_period_months)
    ]
    return config


def split_by_dates(df, config):
    train_test_df = df[
        (df["snapshot_date"] >= config["train_test_start_date"]) &
        (df["snapshot_date"] <= config["train_test_end_date"])
    ].reset_index(drop=True)
    oot_df = df[
        (df["snapshot_date"] >= config["oot_start_date"]) &
        (df["snapshot_date"] <= config["oot_end_date"])
    ].reset_index(drop=True)
    return train_test_df, oot_df


def split_oot_by_month(oot_df, config):
    return [
        oot_df[(oot_df["snapshot_date"] >= start) & (oot_df["snapshot_date"] <= end)].reset_index(drop=True)
        for start, end in config["oot_month_ranges"]
    ]


def split_train_val_test(train_test_df, feature_cols, config, random_state=88):
    X, y = train_test_df[feature_cols], train_test_df["label"]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=1 - config["train_ratio"], random_state=random_state, shuffle=True, stratify=y,
    )
    val_share = config["val_ratio"] / (config["val_ratio"] + config["test_ratio"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=1 - val_share, random_state=random_state, shuffle=True, stratify=y_rest,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])


def get_candidate_grids(scale_pos_weight=1.0):
    return {
        "logistic_regression": {
            "estimator_cls": LogisticRegression,
            "fixed_params": {
                "penalty": "l2", "class_weight": "balanced",
                "max_iter": 1000, "random_state": RANDOM_STATE,
            },
            "param_grid": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
        },
        "xgboost": {
            "estimator_cls": XGBClassifier,
            "fixed_params": {
                "scale_pos_weight": scale_pos_weight, "eval_metric": "logloss",
                "random_state": RANDOM_STATE, "n_jobs": -1,
            },
            "param_grid": [
                {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
                {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1},
                {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
            ],
        },
    }


def evaluate_model(pipeline, X, y, threshold=0.5):
    proba = pipeline.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    auc = float(roc_auc_score(y, proba))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }


def score_distribution(pipeline, X, n_bins=10):
    proba = pipeline.predict_proba(X)[:, 1]
    bin_edges = np.quantile(proba, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = 0.0, 1.0
    bin_counts, _ = np.histogram(proba, bins=bin_edges)
    return {
        "bin_edges": bin_edges.tolist(),
        "bin_proportions": (bin_counts / bin_counts.sum()).tolist(),
    }


def numeric_feature_distribution(values, n_bins=10):
    values = values.dropna().to_numpy()
    bin_edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    bin_counts, _ = np.histogram(values, bins=bin_edges)
    return {
        "type": "numeric",
        "bin_edges": bin_edges.tolist(),
        "bin_proportions": (bin_counts / bin_counts.sum()).tolist(),
    }


def categorical_feature_distribution(values):
    proportions = values.fillna("Unknown").value_counts(normalize=True)
    return {
        "type": "categorical",
        "categories": proportions.index.tolist(),
        "bin_proportions": proportions.to_numpy().tolist(),
    }


def feature_distributions(df, numeric_cols, categorical_cols, n_bins=10):
    distributions = {}
    for col in numeric_cols:
        distributions[col] = numeric_feature_distribution(df[col], n_bins)
    for col in categorical_cols:
        distributions[col] = categorical_feature_distribution(df[col])
    return distributions


def train_and_select_model(train_test_df, oot_df, numeric_cols, categorical_cols, config):
    feature_cols = numeric_cols + categorical_cols

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(train_test_df, feature_cols, config)
    X_oot, y_oot = oot_df[feature_cols], oot_df["label"]
    oot_splits = [
        (f"oot{i + 1}", oot_month_df[feature_cols], oot_month_df["label"])
        for i, oot_month_df in enumerate(split_oot_by_month(oot_df, config))
    ]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    artefacts = {}
    for name, spec in get_candidate_grids(scale_pos_weight).items():
        best_pipeline, best_hp_params, best_val_auc = None, None, -np.inf
        for param_combo in spec["param_grid"]:
            hp_params = {**spec["fixed_params"], **param_combo}
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
                ("classifier", spec["estimator_cls"](**hp_params)),
            ])
            pipeline.fit(X_train, y_train)
            val_auc = evaluate_model(pipeline, X_val, y_val)["auc"]
            if val_auc > best_val_auc:
                best_pipeline, best_hp_params, best_val_auc = pipeline, hp_params, val_auc

        pipeline, hp_params = best_pipeline, best_hp_params

        eval_splits = [
            ("train", X_train, y_train), ("val", X_val, y_val),
            ("test", X_test, y_test), ("oot", X_oot, y_oot),
        ] + oot_splits

        results = {}
        for split_name, X_split, y_split in eval_splits:
            metrics = evaluate_model(pipeline, X_split, y_split)
            results[f"auc_{split_name}"] = metrics["auc"]
            results[f"gini_{split_name}"] = metrics["gini"]
            if split_name != "train":
                results[f"accuracy_{split_name}"] = metrics["accuracy"]
                results[f"precision_{split_name}"] = metrics["precision"]
                results[f"recall_{split_name}"] = metrics["recall"]
                results[f"f1_{split_name}"] = metrics["f1"]
                results[f"confusion_matrix_{split_name}"] = metrics["confusion_matrix"]

        data_stats = {
            "X_train": int(X_train.shape[0]), "y_train": round(float(y_train.mean()), 4),
            "X_val": int(X_val.shape[0]), "y_val": round(float(y_val.mean()), 4),
            "X_test": int(X_test.shape[0]), "y_test": round(float(y_test.mean()), 4),
            "X_oot": int(X_oot.shape[0]), "y_oot": round(float(y_oot.mean()), 4),
        }
        for split_name, X_split, y_split in oot_splits:
            data_stats[f"X_{split_name}"] = int(X_split.shape[0])
            data_stats[f"y_{split_name}"] = round(float(y_split.mean()), 4)

        artefacts[name] = {
            "model": pipeline,
            "model_name": name,
            "label_def": LABEL_DEF,
            "preprocessing_transformers": {},
            "feature_cols": {"numeric": numeric_cols, "categorical": categorical_cols},
            "data_dates": config,
            "data_stats": data_stats,
            "results": results,
            "hp_params": hp_params,
            "training_score_distribution": score_distribution(pipeline, train_test_df[feature_cols]),
            "training_feature_distributions": feature_distributions(train_test_df, numeric_cols, categorical_cols),
            "reference_features": train_test_df[feature_cols].sample(
                n=min(5000, len(train_test_df)), random_state=42
            ).reset_index(drop=True),
        }

    for name, artefact in artefacts.items():
        artefact["model_role"] = MODEL_ROLES[name]

    champion_name = next(name for name, role in MODEL_ROLES.items() if role == "champion")
    return champion_name, artefacts


def save_model_artifact(artefact, model_bank_dir, model_train_date_str):
    os.makedirs(model_bank_dir, exist_ok=True)

    model_version = f"loan_default_{artefact['model_name']}_{model_train_date_str.replace('-', '_')}"
    artefact["model_version"] = model_version

    pkl_path = os.path.join(model_bank_dir, f"{model_version}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artefact, f)

    summary = {k: v for k, v in artefact.items() if k not in ("model", "preprocessing_transformers")}
    with open(os.path.join(model_bank_dir, f"{model_version}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return pkl_path


def run_training(gold_loans_path=GOLD_LOANS_PATH, model_bank_dir=MODEL_BANK_DIR,
                 model_train_date_str=MODEL_TRAIN_DATE_STR, train_test_period_months=TRAIN_TEST_MONTHS,
                 oot_period_months=OOT_MONTHS, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    config = build_config(model_train_date_str, train_test_period_months, oot_period_months,
                          train_ratio, val_ratio, test_ratio)

    df = load_training_data(gold_loans_path)
    numeric_cols, categorical_cols = get_feature_columns(df)
    train_test_df, oot_df = split_by_dates(df, config)
    print(f"loaded {len(df)} rows from {gold_loans_path}; "
          f"train_test={len(train_test_df)} rows ({config['train_test_start_date'].date()} - {config['train_test_end_date'].date()}), "
          f"oot={len(oot_df)} rows ({config['oot_start_date'].date()} - {config['oot_end_date'].date()})")
    print(f"numeric_cols ({len(numeric_cols)}): {numeric_cols}")
    print(f"categorical_cols ({len(categorical_cols)}): {categorical_cols}")

    champion_name, artefacts = train_and_select_model(train_test_df, oot_df, numeric_cols, categorical_cols, config)

    # too large for Airflow logs - still saved in the artefact and *_summary.json, just not echoed
    not_loggable = {"model", "reference_features", "training_score_distribution", "training_feature_distributions"}

    artifact_paths = {}
    for name, artefact in artefacts.items():
        artifact_paths[name] = save_model_artifact(artefact, model_bank_dir, model_train_date_str)
        print(f"--- {name} ({artefact['model_role']}) ---")
        pprint.pprint({k: v for k, v in artefact.items() if k not in not_loggable})
        print(f"saved pkl: {artifact_paths[name]}")
        print(f"saved summary: {artifact_paths[name].replace('.pkl', '_summary.json')}")

    return artifact_paths, champion_name
