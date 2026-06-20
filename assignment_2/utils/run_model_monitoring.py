import argparse

from utils.model_monitoring import run_monitoring


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model monitoring for a snapshot date")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True,
                         help="model artefact filename in model_bank, e.g. loan_default_xgboost_2024_07_01.pkl")
    args = parser.parse_args()

    run_monitoring(args.snapshotdate, args.modelname)
