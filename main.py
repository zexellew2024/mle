import os
import glob
from datetime import datetime

import pyspark

from utils.constants import (
    BRONZE_LMS_DIR, BRONZE_FEAT_DIR, SILVER_LMS_DIR, SILVER_FEAT_DIR,
    GOLD_LABEL_DIR, GOLD_FEAT_DIR, GOLD_INTEGRATED_DIR, MODEL_BANK_DIR,
    START_DATE_STR, END_DATE_STR, DPD_CUTOFF, MOB_CUTOFF
)
from utils.data_processing_bronze_table import process_bronze_table, process_bronze_table_features
from utils.data_processing_silver_table import process_silver_table, process_silver_table_features
from utils.data_processing_gold_table import (
    process_labels_gold_table, process_features_gold_table, feature_label_integration
)
from utils import model_training


def create_spark_session():
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def init_datamart():
    for directory in [BRONZE_LMS_DIR, BRONZE_FEAT_DIR, SILVER_LMS_DIR, SILVER_FEAT_DIR,
                       GOLD_LABEL_DIR, GOLD_FEAT_DIR, GOLD_INTEGRATED_DIR, MODEL_BANK_DIR]:
        os.makedirs(directory, exist_ok=True)


def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))

        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates


def run_datamart_backfill(spark, dates_str_lst):
    for date_str in dates_str_lst:
        process_bronze_table(date_str, BRONZE_LMS_DIR, spark)
        process_bronze_table_features(date_str, spark)

    for date_str in dates_str_lst:
        process_silver_table(date_str, BRONZE_LMS_DIR, SILVER_LMS_DIR, spark)
        process_silver_table_features(date_str, spark)

    for date_str in dates_str_lst:
        process_labels_gold_table(date_str, SILVER_LMS_DIR, GOLD_LABEL_DIR, spark, dpd=DPD_CUTOFF, mob=MOB_CUTOFF)
        process_features_gold_table(date_str, spark)


def build_gold_loans(spark):
    label_files = glob.glob(os.path.join(GOLD_LABEL_DIR, '*'))
    df_label = spark.read.parquet(*label_files)
    print("label_store row count:", df_label.count())

    feature_files = glob.glob(os.path.join(GOLD_FEAT_DIR, '*'))
    df_features = spark.read.parquet(*feature_files)
    print("feature_store row count:", df_features.count())

    return feature_label_integration(df_features, df_label, performance_window_months=MOB_CUTOFF)


if __name__ == "__main__":
    spark = create_spark_session()
    init_datamart()

    dates_str_lst = generate_first_of_month_dates(START_DATE_STR, END_DATE_STR)
    print(dates_str_lst)

    run_datamart_backfill(spark, dates_str_lst)

    gold_loans = build_gold_loans(spark)
    print("gold_loans row count:", gold_loans.count())

    artifact_paths, champion_name = model_training.run_training()
    print("model artifacts saved to:", artifact_paths)
    print("champion model:", champion_name)
