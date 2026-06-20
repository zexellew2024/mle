import json
import os
import pickle
import resource
import time

import pandas as pd

from utils.constants import GOLD_FEAT_DIR, GOLD_PREDICTIONS_DIR, MODEL_BANK_DIR


def load_model_artefact(model_filename, model_bank_dir=MODEL_BANK_DIR):
    with open(os.path.join(model_bank_dir, model_filename), "rb") as f:
        return pickle.load(f)


def run_inference(snapshot_date_str, model_filename, model_bank_dir=MODEL_BANK_DIR,
                   gold_feat_dir=GOLD_FEAT_DIR, output_dir=GOLD_PREDICTIONS_DIR):
    artefact = load_model_artefact(model_filename, model_bank_dir)

    partition_name = "gold_feature_store_" + snapshot_date_str.replace("-", "_") + ".parquet"
    df = pd.read_parquet(os.path.join(gold_feat_dir, partition_name))

    feature_cols = artefact["feature_cols"]["numeric"] + artefact["feature_cols"]["categorical"]

    rusage_before = resource.getrusage(resource.RUSAGE_SELF)
    start_time = time.perf_counter()
    predictions = artefact["model"].predict_proba(df[feature_cols])[:, 1]
    latency_sec = time.perf_counter() - start_time
    rusage_after = resource.getrusage(resource.RUSAGE_SELF)

    n_scored = int(len(df))

    output_df = pd.DataFrame({
        "Customer_ID": df["Customer_ID"],
        "snapshot_date": pd.to_datetime(df["snapshot_date"]).dt.strftime("%Y-%m-%d"),
        "model_name": artefact["model_name"],
        "model_version": artefact["model_version"],
        "model_predictions": predictions,
    })

    model_version = artefact["model_version"]
    out_dir = os.path.join(output_dir, model_version)
    os.makedirs(out_dir, exist_ok=True)

    partition_name = f"{model_version}_predictions_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = os.path.join(out_dir, partition_name)
    output_df.to_parquet(filepath)
    print("saved to:", filepath)

    meta = {
        "snapshot_date": snapshot_date_str,
        "model_name": artefact["model_name"],
        "model_version": model_version,
        "n_scored": n_scored,
        "latency_sec": latency_sec,
        "throughput_rps": (n_scored / latency_sec) if latency_sec > 0 else None,
        "cpu_time_sec": (rusage_after.ru_utime + rusage_after.ru_stime) - (rusage_before.ru_utime + rusage_before.ru_stime),
        "peak_memory_mb": rusage_after.ru_maxrss / 1024,
        "error_rate": float(output_df["model_predictions"].isnull().mean()) if n_scored else 0.0,
    }
    meta_filename = f"{model_version}_predictions_" + snapshot_date_str.replace("-", "_") + "_meta.json"
    meta_filepath = os.path.join(out_dir, meta_filename)
    with open(meta_filepath, "w") as f:
        json.dump(meta, f, indent=2)
    print("saved to:", meta_filepath)

    return filepath
