import glob
import json
import os
import pickle
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from dateutil.relativedelta import relativedelta
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from utils.constants import (
    ACCURACY_GREEN, ACCURACY_YELLOW_HIGH, ACCURACY_YELLOW_LOW, ALERT_LEVEL_COLOR, ALERT_LEVEL_RANK,
    ATTRIBUTES_COLS, AUC_GREEN, AUC_YELLOW_HIGH, AUC_YELLOW_LOW, CHALLENGER_AUC_UPLIFT_THRESHOLD,
    CLICKSTREAM_COLS, CSI_PVALUE_GREEN, CSI_PVALUE_YELLOW_LOW, FINANCIALS_COLS,
    GOLD_FEAT_DIR, GOLD_LABEL_DIR, GOLD_MONITORING_DIR, GOLD_PREDICTIONS_DIR,
    MISSING_FEATURE_ALERT_THRESHOLD, MOB_CUTOFF, MODEL_BANK_DIR, MODEL_TRAIN_DATE_STR, PSI_GREEN,
    PSI_YELLOW_HIGH, PSI_YELLOW_LOW, RECALL_GREEN, RECALL_YELLOW_HIGH, RECALL_YELLOW_LOW,
)


def load_model_artefact(model_filename, model_bank_dir=MODEL_BANK_DIR):
    with open(os.path.join(model_bank_dir, model_filename), "rb") as f:
        return pickle.load(f)


def _run_drift_report(reference_df, current_df, feature_cols, stattest):
    report = Report(metrics=[ColumnDriftMetric(column_name=col, stattest=stattest) for col in feature_cols])
    report.run(reference_data=reference_df, current_data=current_df)
    return {m["result"]["column_name"]: m["result"]["drift_score"] for m in report.as_dict()["metrics"]}


def compute_top_shap_features(pipeline, ref_features, cur_features, feature_cols, categorical_cols=(), top_n=5):
    # SHAP's Independent masker needs a purely numeric matrix (it calls
    # np.isclose internally), but Occupation/Age_group are raw strings the
    # pipeline's own ColumnTransformer encodes. Represent them as category
    # codes for SHAP, and map back to strings inside predict_pos so the
    # pipeline still sees the raw categories it was trained on.
    categories = {
        col: pd.concat([ref_features[col], cur_features[col]]).astype("category").cat.categories
        for col in categorical_cols
    }

    def to_numeric(df):
        df = df[feature_cols].copy()
        for col in categorical_cols:
            df[col] = pd.Categorical(df[col], categories=categories[col]).codes.astype(float)
        return df

    def predict_pos(X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_cols)
        X = X.copy()
        for col in categorical_cols:
            codes = X[col].round().astype(int)
            X[col] = [categories[col][c] if 0 <= c < len(categories[col]) else np.nan for c in codes]
        return pipeline.predict_proba(X)[:, 1]

    ref_numeric = to_numeric(ref_features)
    cur_numeric = to_numeric(cur_features)

    background = shap.maskers.Independent(
        ref_numeric.sample(min(100, len(ref_numeric)), random_state=42)
    )
    explainer = shap.Explainer(predict_pos, background, feature_names=feature_cols)
    sample = cur_numeric.sample(min(200, len(cur_numeric)), random_state=42)
    shap_values = explainer(sample)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    return [
        {"feature": feature_cols[i], "mean_abs_shap": round(float(mean_abs[i]), 6)}
        for i in top_idx
    ]


def _update_alert_level(current_level, new_level):
    level_priority = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    return new_level if level_priority[new_level] > level_priority[current_level] else current_level


def evaluate_performance_alerts(row):
    """Evaluate recall/accuracy/auc (P0/P1/P2) against Green/Yellow/Red zones. Only meaningful
    once a label has matured for this snapshot - returns GREEN/"" when label_available is False."""
    alert_reasons = []
    highest_alert_level = "GREEN"

    if row["label_available"] and not np.isnan(row["recall"]):
        if row["recall"] >= RECALL_GREEN:
            alert_reasons.append(f"recall {row['recall']:.3f} >= {RECALL_GREEN} [GREEN]")
        elif row["recall"] >= RECALL_YELLOW_LOW:
            alert_reasons.append(f"recall {row['recall']:.3f} in [{RECALL_YELLOW_LOW}, {RECALL_YELLOW_HIGH}] [YELLOW]")
            highest_alert_level = _update_alert_level(highest_alert_level, "YELLOW")
        else:
            alert_reasons.append(f"recall {row['recall']:.3f} < {RECALL_YELLOW_LOW} [RED]")
            highest_alert_level = _update_alert_level(highest_alert_level, "RED")

    if row["label_available"] and not np.isnan(row["accuracy"]):
        if row["accuracy"] >= ACCURACY_GREEN:
            alert_reasons.append(f"accuracy {row['accuracy']:.3f} >= {ACCURACY_GREEN} [GREEN]")
        elif row["accuracy"] >= ACCURACY_YELLOW_LOW:
            alert_reasons.append(f"accuracy {row['accuracy']:.3f} in [{ACCURACY_YELLOW_LOW}, {ACCURACY_YELLOW_HIGH}] [YELLOW]")
            highest_alert_level = _update_alert_level(highest_alert_level, "YELLOW")
        else:
            alert_reasons.append(f"accuracy {row['accuracy']:.3f} < {ACCURACY_YELLOW_LOW} [RED]")
            highest_alert_level = _update_alert_level(highest_alert_level, "RED")

    if row["label_available"] and not np.isnan(row["auc"]):
        if row["auc"] >= AUC_GREEN:
            alert_reasons.append(f"auc {row['auc']:.3f} >= {AUC_GREEN} [GREEN]")
        elif row["auc"] >= AUC_YELLOW_LOW:
            alert_reasons.append(f"auc {row['auc']:.3f} in [{AUC_YELLOW_LOW}, {AUC_YELLOW_HIGH}] [YELLOW]")
            highest_alert_level = _update_alert_level(highest_alert_level, "YELLOW")
        else:
            alert_reasons.append(f"auc {row['auc']:.3f} < {AUC_YELLOW_LOW} [RED]")
            highest_alert_level = _update_alert_level(highest_alert_level, "RED")

    return highest_alert_level, "; ".join(alert_reasons)


def evaluate_feature_alerts(row):
    """Evaluate drift & data-quality checks against Green/Yellow/Red zones. Available immediately
    at scoring time - doesn't need a label, unlike evaluate_performance_alerts."""
    alert_reasons = []
    highest_alert_level = "GREEN"

    if not np.isnan(row["psi_score"]):
        if row["psi_score"] <= PSI_GREEN:
            alert_reasons.append(f"psi {row['psi_score']:.3f} <= {PSI_GREEN} [GREEN]")
        elif row["psi_score"] <= PSI_YELLOW_HIGH:
            alert_reasons.append(f"psi {row['psi_score']:.3f} in ({PSI_YELLOW_LOW}, {PSI_YELLOW_HIGH}] [YELLOW]")
            highest_alert_level = _update_alert_level(highest_alert_level, "YELLOW")
        else:
            alert_reasons.append(f"psi {row['psi_score']:.3f} > {PSI_YELLOW_HIGH} [RED]")
            highest_alert_level = _update_alert_level(highest_alert_level, "RED")

    if not np.isnan(row["min_csi_pvalue"]):
        if row["min_csi_pvalue"] > CSI_PVALUE_GREEN:
            alert_reasons.append(f"min_csi_pvalue {row['min_csi_pvalue']:.6f} > {CSI_PVALUE_GREEN} [GREEN]")
        elif row["min_csi_pvalue"] > CSI_PVALUE_YELLOW_LOW:
            alert_reasons.append(f"min_csi_pvalue {row['min_csi_pvalue']:.6f} in ({CSI_PVALUE_YELLOW_LOW}, {CSI_PVALUE_GREEN}] [YELLOW]")
            highest_alert_level = _update_alert_level(highest_alert_level, "YELLOW")
        else:
            alert_reasons.append(f"min_csi_pvalue {row['min_csi_pvalue']:.6f} <= {CSI_PVALUE_YELLOW_LOW} [RED]")
            highest_alert_level = _update_alert_level(highest_alert_level, "RED")

    for source in ("attributes", "financials", "clickstream"):
        pct_missing = row[f"pct_missing_{source}"]
        if pct_missing >= MISSING_FEATURE_ALERT_THRESHOLD:
            alert_reasons.append(f"pct_missing_{source} {pct_missing:.3f} >= {MISSING_FEATURE_ALERT_THRESHOLD} [RED]")
            highest_alert_level = _update_alert_level(highest_alert_level, "RED")

    if row["error_rate"] > 0:
        alert_reasons.append(f"error_rate {row['error_rate']:.3f} > 0 [RED]")
        highest_alert_level = _update_alert_level(highest_alert_level, "RED")

    return highest_alert_level, "; ".join(alert_reasons)


def run_monitoring(snapshot_date_str, model_filename, model_bank_dir=MODEL_BANK_DIR,
                    gold_feat_dir=GOLD_FEAT_DIR, predictions_dir=GOLD_PREDICTIONS_DIR,
                    gold_label_dir=GOLD_LABEL_DIR, output_dir=GOLD_MONITORING_DIR):
    artefact = load_model_artefact(model_filename, model_bank_dir)
    model_version = artefact["model_version"]
    model_name = artefact["model_name"]
    numeric_cols = artefact["feature_cols"]["numeric"]
    categorical_cols = artefact["feature_cols"]["categorical"]
    feature_cols = numeric_cols + categorical_cols

    partition_name = "gold_feature_store_" + snapshot_date_str.replace("-", "_") + ".parquet"
    df_features = pd.read_parquet(os.path.join(gold_feat_dir, partition_name))
    pct_missing_by_source = {
        source: float(df_features[cols].isnull().any(axis=1).mean())
        for source, cols in [("attributes", ATTRIBUTES_COLS), ("financials", FINANCIALS_COLS),
                             ("clickstream", CLICKSTREAM_COLS)]
    }

    ref_features = artefact["reference_features"]
    cur_features = df_features[feature_cols]

    # Evidently raises instead of scoring drift for a column that's entirely null in either
    # dataset (e.g. clickstream for a cohort the panel doesn't cover) - exclude those; the
    # pct_missing_* checks in evaluate_feature_alerts already flag them separately.
    drift_cols = [
        col for col in feature_cols
        if cur_features[col].notna().any() and ref_features[col].notna().any()
    ]
    csi_scores = _run_drift_report(ref_features, cur_features, drift_cols, stattest="psi")

    # chisquare is only valid for categorical columns in Evidently; numeric
    # columns need a numeric significance test (ks = Kolmogorov-Smirnov).
    numeric_drift_cols = [col for col in drift_cols if col in numeric_cols]
    categorical_drift_cols = [col for col in drift_cols if col in categorical_cols]
    csi_pvalues = {
        **_run_drift_report(ref_features, cur_features, numeric_drift_cols, stattest="ks"),
        **_run_drift_report(ref_features, cur_features, categorical_drift_cols, stattest="chisquare"),
    }
    max_csi = max(csi_scores.values()) if csi_scores else np.nan
    mean_csi = float(np.mean(list(csi_scores.values()))) if csi_scores else np.nan
    min_csi_pvalue = min(csi_pvalues.values()) if csi_pvalues else np.nan
    top5_shap = compute_top_shap_features(artefact["model"], ref_features, cur_features, feature_cols, categorical_cols)

    partition_name = f"{model_version}_predictions_" + snapshot_date_str.replace("-", "_") + ".parquet"
    pred_dir = os.path.join(predictions_dir, model_version)
    df_preds = pd.read_parquet(os.path.join(pred_dir, partition_name))

    ref_score_df = pd.DataFrame({"model_predictions": artefact["model"].predict_proba(ref_features)[:, 1]})
    score_report = Report(metrics=[ColumnDriftMetric(column_name="model_predictions", stattest="psi")])
    score_report.run(reference_data=ref_score_df, current_data=df_preds[["model_predictions"]])
    psi_score = score_report.as_dict()["metrics"][0]["result"]["drift_score"]

    meta_filename = f"{model_version}_predictions_" + snapshot_date_str.replace("-", "_") + "_meta.json"
    with open(os.path.join(pred_dir, meta_filename)) as f:
        meta = json.load(f)

    row = {
        "snapshot_date": snapshot_date_str,
        "model_name": model_name,
        "model_version": model_version,
        "n_scored": int(len(df_preds)),
        "pct_missing_attributes": pct_missing_by_source["attributes"],
        "pct_missing_financials": pct_missing_by_source["financials"],
        "pct_missing_clickstream": pct_missing_by_source["clickstream"],
        "psi_score": psi_score,
        "max_csi": max_csi,
        "mean_csi": mean_csi,
        "min_csi_pvalue": min_csi_pvalue,
        "latency_sec": meta["latency_sec"],
        "throughput_rps": meta["throughput_rps"],
        "cpu_time_sec": meta["cpu_time_sec"],
        "peak_memory_mb": meta["peak_memory_mb"],
        "error_rate": meta["error_rate"],
        "top5_shap": json.dumps(top5_shap),
        "label_available": False,
        "auc": np.nan,
        "gini": np.nan,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
    }

    # a label matures MOB_CUTOFF months after its feature snapshot, so it lives in that later
    # gold_label_store partition (mob == MOB_CUTOFF) if it exists yet
    label_snapshot_date = (
        datetime.strptime(snapshot_date_str, "%Y-%m-%d") + relativedelta(months=MOB_CUTOFF)
    ).strftime("%Y-%m-%d")
    label_partition = os.path.join(gold_label_dir, "gold_label_store_" + label_snapshot_date.replace("-", "_") + ".parquet")

    df_labels = (
        pd.read_parquet(label_partition, columns=["Customer_ID", "label"])
        if os.path.exists(label_partition) else pd.DataFrame(columns=["Customer_ID", "label"])
    )

    if len(df_labels) > 0:
        df_joined = df_preds.merge(df_labels, on="Customer_ID", how="inner")
        y_true = df_joined["label"]
        y_proba = df_joined["model_predictions"]
        y_pred = (y_proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y_true, y_proba))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        accuracy = float(accuracy_score(y_true, y_pred))
        row.update({
            "label_available": True,
            "auc": auc,
            "gini": 2 * auc - 1,
            "accuracy": accuracy,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": recall,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        })

    row["performance_alert_level"], row["performance_alert_reasons"] = evaluate_performance_alerts(row)
    row["feature_alert_level"], row["feature_alert_reasons"] = evaluate_feature_alerts(row)

    out_df = pd.DataFrame([row])
    out_dir = os.path.join(output_dir, model_version)
    os.makedirs(out_dir, exist_ok=True)
    date_slug = snapshot_date_str.replace("-", "_")

    viz_report = Report(metrics=[DataDriftPreset(stattest="psi")])
    viz_report.run(reference_data=ref_features[drift_cols], current_data=cur_features[drift_cols])
    html_path = os.path.join(out_dir, f"{model_version}_monitoring_{date_slug}.html")
    viz_report.save_html(html_path)
    print("report saved to:", html_path)

    parquet_path = os.path.join(out_dir, f"{model_version}_monitoring_{date_slug}.parquet")
    out_df.to_parquet(parquet_path)
    print("metrics saved to:", parquet_path)

    return parquet_path


def load_monitoring_history(model_version, monitoring_dir=GOLD_MONITORING_DIR):
    """Concatenate every per-day monitoring parquet on disk for a model_version, oldest first."""
    model_dir = os.path.join(monitoring_dir, model_version)
    files = sorted(glob.glob(os.path.join(model_dir, f"{model_version}_monitoring_*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df.sort_values("snapshot_date").reset_index(drop=True)


def _shap_long_frame(df):
    """Explode the per-day top5_shap JSON column into a long (snapshot_date, feature, mean_abs_shap) frame."""
    return pd.DataFrame([
        {"snapshot_date": row["snapshot_date"], "feature": entry["feature"], "mean_abs_shap": entry["mean_abs_shap"]}
        for _, row in df.iterrows()
        if isinstance(row["top5_shap"], str)
        for entry in json.loads(row["top5_shap"])
    ])


def plot_monitoring_summary(model_versions, asof_date_str, labels=None, monitoring_dir=GOLD_MONITORING_DIR,
                             output_dir=None):
    """Render one multi-panel figure (performance, drift, data quality, alert level, SHAP feature
    importance) across all snapshot dates monitored so far, optionally overlaying several
    model_versions (e.g. champion vs challenger) for comparison."""
    if isinstance(model_versions, str):
        model_versions = [model_versions]
    labels = list(labels) if labels else list(model_versions)
    asof_ts = pd.Timestamp(asof_date_str)
    train_ts = pd.Timestamp(MODEL_TRAIN_DATE_STR)

    histories = {}
    for version in model_versions:
        df = load_monitoring_history(version, monitoring_dir)
        df = df[(df["snapshot_date"] >= train_ts) & (df["snapshot_date"] <= asof_ts)]
        if not df.empty:
            histories[version] = df

    if not histories:
        raise ValueError(f"No monitoring history found for {model_versions} as of {asof_date_str}")

    linestyles = ["-", "--", ":", "-."]
    fig, (ax_perf, ax_drift, ax_missing, ax_perf_alert, ax_feat_alert, ax_shap) = plt.subplots(
        6, 1, figsize=(14, 26), sharex=True)

    # (metric, color, green cutoff, red cutoff) - RAG thresholds from utils/constants.py
    perf_metrics = [
        ("auc", "tab:blue", AUC_GREEN, AUC_YELLOW_LOW),
        ("accuracy", "tab:green", ACCURACY_GREEN, ACCURACY_YELLOW_LOW),
        ("recall", "tab:orange", RECALL_GREEN, RECALL_YELLOW_LOW),
    ]

    for i, (version, df) in enumerate(histories.items()):
        ls = linestyles[i % len(linestyles)]
        label_name = labels[model_versions.index(version)]
        scored = df[df["label_available"]]
        for metric, color, _, _ in perf_metrics:
            if scored[metric].notna().any():
                ax_perf.plot(scored["snapshot_date"], scored[metric], marker="o", linestyle=ls,
                             color=color, label=f"{label_name} {metric}")
        ax_drift.plot(df["snapshot_date"], df["psi_score"], marker="o", linestyle=ls,
                      color="tab:red", label=f"{label_name} psi_score (prediction drift)")

    # pct_missing_* is model-independent (straight from gold_feature_store) - plot it once
    df0 = next(iter(histories.values()))
    missing_metrics = [("pct_missing_attributes", "tab:blue"), ("pct_missing_financials", "tab:green"),
                       ("pct_missing_clickstream", "tab:brown")]
    for metric, color in missing_metrics:
        ax_missing.plot(df0["snapshot_date"], df0[metric], marker="o", color=color, label=metric)

    # max_csi reflects the shared input features, not the model - plot it once when models agree,
    # falling back to per-model lines if a model's own feature set makes it diverge
    aligned_csi = pd.DataFrame({v: df.set_index("snapshot_date")["max_csi"] for v, df in histories.items()})
    if aligned_csi.nunique(axis=1, dropna=True).le(1).all():
        df0 = next(iter(histories.values()))
        ax_drift.plot(df0["snapshot_date"], df0["max_csi"], marker="s", linestyle="-",
                      color="tab:purple", label="max_csi (feature drift, shared across models)")
    else:
        for i, (version, df) in enumerate(histories.items()):
            ls = linestyles[i % len(linestyles)]
            label_name = labels[model_versions.index(version)]
            ax_drift.plot(df["snapshot_date"], df["max_csi"], marker="s", linestyle=ls,
                          color="tab:purple", label=f"{label_name} max_csi (feature drift)")

    for metric, color, green, red in perf_metrics:
        ax_perf.axhline(green, color=color, linestyle=":", alpha=0.5, linewidth=1)
        ax_perf.axhline(red, color=color, linestyle="--", alpha=0.5, linewidth=1)
    ax_drift.axhline(PSI_GREEN, color="tab:red", linestyle=":", alpha=0.5, linewidth=1)
    ax_drift.axhline(PSI_YELLOW_HIGH, color="tab:red", linestyle="--", alpha=0.5, linewidth=1)

    ax_perf.set_ylabel("Performance")
    ax_perf.set_title("Model performance over time (scored snapshots only)")
    ax_perf.text(0.01, 0.02, "dotted = GREEN cutoff, dashed = RED cutoff", transform=ax_perf.transAxes,
                 fontsize=7, va="bottom", ha="left", style="italic")
    ax_perf.legend(fontsize=8, loc="upper left")
    ax_perf.grid(alpha=0.3)

    ax_drift.set_ylabel("Drift (PSI / CSI)")
    ax_drift.set_title("Prediction drift (per model) & feature drift (shared) over time")
    ax_drift.legend(fontsize=8, loc="upper left")
    ax_drift.grid(alpha=0.3)

    ax_missing.set_ylabel("Pct rows with a missing value")
    ax_missing.set_title("Missing values by data source over time")
    ax_missing.legend(fontsize=8, loc="upper left")
    ax_missing.grid(alpha=0.3)

    def plot_alert_panel(ax, column, title):
        for i, (version, df) in enumerate(histories.items()):
            label_name = labels[model_versions.index(version)]
            ranks = df[column].map(ALERT_LEVEL_RANK)
            colors = df[column].map(ALERT_LEVEL_COLOR).fillna("tab:gray")
            offset = (i - (len(histories) - 1) / 2) * 0.08
            ax.scatter(df["snapshot_date"], ranks + offset, c=colors, s=60,
                       marker="o" if i % 2 == 0 else "s", label=label_name)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["GREEN", "YELLOW", "RED"])
        ax.set_ylim(-0.5, 2.5)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    plot_alert_panel(ax_perf_alert, "performance_alert_level", "Performance alert level over time (recall / accuracy / auc)")
    plot_alert_panel(ax_feat_alert, "feature_alert_level", "Feature/data alert level over time (drift & data quality)")

    shap_frames = {version: _shap_long_frame(df) for version, df in histories.items()}
    shap_frames = {version: frame for version, frame in shap_frames.items() if not frame.empty}
    all_features = sorted({f for frame in shap_frames.values() for f in frame["feature"].unique()})
    shap_cmap = plt.get_cmap("tab20")
    feature_color = {f: shap_cmap(i % 20) for i, f in enumerate(all_features)}
    for i, (version, frame) in enumerate(shap_frames.items()):
        ls = linestyles[i % len(linestyles)]
        label_name = labels[model_versions.index(version)]
        for feature in sorted(frame["feature"].unique()):
            sub = frame[frame["feature"] == feature].sort_values("snapshot_date")
            ax_shap.plot(sub["snapshot_date"], sub["mean_abs_shap"], marker="o", linestyle=ls,
                         color=feature_color[feature], label=f"{label_name} {feature}")
    ax_shap.set_ylabel("mean |SHAP|")
    ax_shap.set_title("Top-5 SHAP feature importance over time (gaps = feature fell out of that month's top 5)")
    ax_shap.legend(fontsize=6, loc="upper left", ncol=2, bbox_to_anchor=(1.01, 1))
    ax_shap.grid(alpha=0.3)
    ax_shap.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    # mirror date labels onto the top panel too (sharex only labels the bottom one by default)
    ax_perf.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_perf.tick_params(axis="x", which="both", top=True, labeltop=True)
    plt.setp(ax_perf.get_xticklabels(), rotation=45, ha="left")

    fig.suptitle(f"Monitoring summary as of {asof_date_str}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.8, 0.97])

    output_dir = output_dir or os.path.join(monitoring_dir, "summary")
    os.makedirs(output_dir, exist_ok=True)
    asof_slug = asof_date_str.replace("-", "_")
    png_path = os.path.join(output_dir, f"monitoring_summary_{asof_slug}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print("summary saved to:", png_path)
    return png_path


def run_monitoring_summary(snapshot_date_str, model_filenames, model_bank_dir=MODEL_BANK_DIR,
                            monitoring_dir=GOLD_MONITORING_DIR):
    model_versions = []
    labels = []
    for filename in model_filenames:
        artefact = load_model_artefact(filename, model_bank_dir)
        model_versions.append(artefact["model_version"])
        labels.append(artefact["model_name"])
    return plot_monitoring_summary(model_versions, snapshot_date_str, labels=labels, monitoring_dir=monitoring_dir)


def compare_champion_challenger(champion_results, challenger_results):
    uplift = float(challenger_results["auc_oot"] - champion_results["auc_oot"])
    return {
        "challenger_auc_uplift": uplift,
        "ab_test_recommended": bool(uplift >= CHALLENGER_AUC_UPLIFT_THRESHOLD),
    }


def run_comparison(snapshot_date_str, champion_filename, challenger_filename, model_bank_dir=MODEL_BANK_DIR,
                    output_dir=GOLD_MONITORING_DIR):
    champion = load_model_artefact(champion_filename, model_bank_dir)
    challenger = load_model_artefact(challenger_filename, model_bank_dir)

    row = {
        "snapshot_date": snapshot_date_str,
        "champion_model_version": champion["model_version"],
        "challenger_model_version": challenger["model_version"],
        "champion_auc_oot": champion["results"]["auc_oot"],
        "challenger_auc_oot": challenger["results"]["auc_oot"],
        **compare_champion_challenger(champion["results"], challenger["results"]),
    }

    out_df = pd.DataFrame([row])
    out_dir = os.path.join(output_dir, "champion_vs_challenger")
    os.makedirs(out_dir, exist_ok=True)
    partition_name = "champion_vs_challenger_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = os.path.join(out_dir, partition_name)
    out_df.to_parquet(filepath)
    print("saved to:", filepath)

    return filepath
