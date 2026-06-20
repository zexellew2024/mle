import argparse

from utils.model_monitoring import run_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare champion vs challenger model for a snapshot date")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--champion", type=str, required=True,
                         help="champion model artefact filename in model_bank, e.g. loan_default_logistic_regression_2024_07_01.pkl")
    parser.add_argument("--challenger", type=str, required=True,
                         help="challenger model artefact filename in model_bank, e.g. loan_default_xgboost_2024_07_01.pkl")
    args = parser.parse_args()

    run_comparison(args.snapshotdate, args.champion, args.challenger)
