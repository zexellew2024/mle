import pyspark.sql.functions as F
from pyspark.sql.functions import col, expr, date_format
from pyspark.sql.types import StringType, IntegerType

from utils.constants import GOLD_FEAT_DIR, SILVER_FEAT_DIR, FEATURE_FILENAMES, GOLD_LOANS_PATH, MOB_CUTOFF


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    df = df.filter(col("mob") == mob)

    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd) + 'dpd_' + str(mob) + 'mob').cast(StringType()))

    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    partition_name = "gold_label_store_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_features_gold_table(snapshot_date_str, spark):
    dataframes = {}
    for feat_file_name in FEATURE_FILENAMES:
        partition_name = f"silver_{feat_file_name}" + snapshot_date_str.replace('-', '_') + '.parquet'
        filepath = SILVER_FEAT_DIR + partition_name
        df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', df.count())

        # For the interest of predicting new customers, these features are removed as they may leak the outcome
        leakage_cols = [
            'Num_of_Delayed_Payment',
            'Delay_from_due_date',
            'Outstanding_Debt',
            'Payment_of_Min_Amount',
        ]
        df = df.drop(*leakage_cols)

        dataframes[feat_file_name] = df

    df_list = list(dataframes.values())

    # feature_clickstream's customer panel doesn't cover the new customers
    # introduced from 2024-07 onward, so it must be a left join (missing
    # fe_* columns get median-imputed by the model pipeline). attributes and
    # financials always share the same customer set, so stay inner.
    df_gold = df_list[0]
    for i in range(1, len(df_list)):
        join_type = "left" if FEATURE_FILENAMES[i] == "feature_clickstream" else "inner"
        df_gold = df_gold.join(df_list[i], on=["Customer_ID", "snapshot_date"], how=join_type)

    print(f'Final gold table row count after joins: {df_gold.count()}')

    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = GOLD_FEAT_DIR + partition_name
    df_gold.write.mode("overwrite").parquet(filepath)
    print(f'saved to: {filepath}')

    return df_gold


def feature_label_integration(df_features, df_label, performance_window_months=MOB_CUTOFF):
    # Shift the label's snapshot_date back to the feature snapshot_date it was observed at
    df_label_aligned = df_label.withColumn(
        "snapshot_date",
        date_format(expr(f"add_months(to_date(snapshot_date, 'yyyy-MM-dd'), -{performance_window_months})"), "yyyy-MM-dd")
    )

    gold_loans = df_features.join(df_label_aligned,
                                   on=["Customer_ID", "snapshot_date"],
                                   how="inner")

    print(f'Final gold_loans row count: {gold_loans.count()}')

    gold_loans.write.mode('overwrite').parquet(GOLD_LOANS_PATH)
    print(f'saved to: {GOLD_LOANS_PATH}')

    return gold_loans
