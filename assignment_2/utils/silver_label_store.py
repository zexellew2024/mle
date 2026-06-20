import argparse

import pyspark

from utils.constants import BRONZE_LMS_DIR, SILVER_LMS_DIR
from utils.data_processing_silver_table import process_silver_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run silver label store processing for a snapshot date")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_silver_table(args.snapshotdate, BRONZE_LMS_DIR, SILVER_LMS_DIR, spark)

    spark.stop()
