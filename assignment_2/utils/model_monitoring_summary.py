import argparse

from utils.model_monitoring import run_monitoring_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a cross-day monitoring summary visualization")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, nargs="+",
                         help="one or more model artefact filenames in model_bank to overlay, e.g. "
                              "loan_default_logistic_regression_2024_07_01.pkl loan_default_xgboost_2024_07_01.pkl")
    args = parser.parse_args()

    run_monitoring_summary(args.snapshotdate, args.modelname)
