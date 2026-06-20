import argparse

from utils.model_inference import run_inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference for a snapshot date")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True,
                         help="model artefact filename in model_bank, e.g. loan_default_xgboost_2024_07_01.pkl")
    args = parser.parse_args()

    run_inference(args.snapshotdate, args.modelname)
