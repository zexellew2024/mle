import argparse

import pyspark

from utils.data_processing_gold_table import process_features_gold_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gold feature store processing for a snapshot date")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_features_gold_table(args.snapshotdate, spark)

    spark.stop()
