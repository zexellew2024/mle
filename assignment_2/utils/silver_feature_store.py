import argparse

import pyspark

from utils.data_processing_silver_table import process_silver_table_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run silver feature store processing for a snapshot date")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_silver_table_features(args.snapshotdate, spark)

    spark.stop()
