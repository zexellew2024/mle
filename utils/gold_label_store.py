import argparse

import pyspark

from utils.constants import DPD_CUTOFF, GOLD_LABEL_DIR, MOB_CUTOFF, SILVER_LMS_DIR
from utils.data_processing_gold_table import process_labels_gold_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gold label store processing for a snapshot date")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_labels_gold_table(args.snapshotdate, SILVER_LMS_DIR, GOLD_LABEL_DIR, spark, dpd=DPD_CUTOFF, mob=MOB_CUTOFF)

    spark.stop()
