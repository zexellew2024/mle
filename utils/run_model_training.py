import argparse

import main as a2_main
from utils import model_training as model_training_module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild gold_loans and train/select the loan default model")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD (model_train_date)")
    args = parser.parse_args()

    spark = a2_main.create_spark_session()
    gold_loans = a2_main.build_gold_loans(spark)
    print("gold_loans row count:", gold_loans.count())
    spark.stop()

    artifact_paths, champion_name = model_training_module.run_training(model_train_date_str=args.snapshotdate)
    print("model artifacts saved to:", artifact_paths)
    print("champion model:", champion_name)
