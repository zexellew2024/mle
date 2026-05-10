import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse
from functools import reduce

from pyspark.sql.functions import col, row_number, expr, to_date, date_format, coalesce, lit, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql import Window
from pyspark.ml.feature import Imputer

from utils.constants import GOLD_FEAT_DIR, SILVER_FEAT_DIR, FEATURE_FILENAMES, GOLD_INTEGRATED_DIR


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_features_gold_table(snapshot_date_str, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    return_dict = {}

    dataframes = {}
    for feat_file_name in FEATURE_FILENAMES:
    
        # connect to silver table
        partition_name = f"silver_{feat_file_name}" + snapshot_date_str.replace('-','_') + '.parquet'
        filepath = SILVER_FEAT_DIR + partition_name
        df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', df.count())

        # For the interest of predicting new customers, following features to be removed, as these might lead to data leakage.
        leakage_cols = [
            'Num_of_Delayed_Payment',    # How many payments were late
            'Delay_from_due_date',       # Days late 
            'Outstanding_Debt',          # Debt at default time
            'Payment_of_Min_Amount',     # Whether minimum was paid
        ]

        df = df.drop(*leakage_cols)
        
        dataframes[feat_file_name] = df

    df_list = list(dataframes.values())

    # Start with the first dataframe and join with the rest
    df_gold = df_list[0]
    for i in range(1, len(df_list)):
        df_gold = df_gold.join(
            df_list[i], 
            on=["Customer_ID", "snapshot_date"], 
            how="inner"  
    )

    print(f'Final gold table row count after joins: {df_gold.count()}')
    
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = GOLD_FEAT_DIR + partition_name
    df_gold.write.mode("overwrite").parquet(filepath)
    print(f'saved to: {filepath}')
    
    return df_gold

def feature_label_integration(df_features, df_label, performance_window_months=6):    
    df_features_aligned = df_features.withColumn(
        "snapshot_date", 
        date_format(expr(f"add_months(to_date(snapshot_date, 'yyyy-MM-dd'), {performance_window_months})"), "yyyy-MM-dd")
    )

    gold_loans = df_features_aligned.join(df_label,
                                          on=["Customer_ID", "snapshot_date"],
                                          how="inner")

    print(f'Intermediate gold_loans row count: {gold_loans.count()}')
    
    # Impute null with median for numeric columns
    cols_to_impute = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                      'Changed_Credit_Limit', 'Amount_invested_monthly', 'Monthly_Balance',
                      'Credit_Mix_Label', 'Spending_Level', 'Payment_Value', 'Annual_Income']
    imputer = Imputer(strategy="median", inputCols=cols_to_impute, outputCols=cols_to_impute)
    gold_loans = imputer.fit(gold_loans).transform(gold_loans)

    # Impute mode for occupation, then convert to binary
    distinct_occupations = gold_loans.filter(col('Occupation').isNotNull()) \
                                     .select('Occupation') \
                                     .distinct() \
                                     .rdd.flatMap(lambda x: x) \
                                     .collect()
    
    # Create a column for each occupation (1 if matches, 0 otherwise)
    for occupation in distinct_occupations:
        # Create a valid column name (replace spaces/special chars with underscore)
        col_name = f"is_{occupation.replace(' ', '_').replace('-', '_')}"
        gold_loans = gold_loans.withColumn(col_name, 
                                           when(col('Occupation') == occupation, 1)
                                           .otherwise(0)
                                           .cast(IntegerType()))
    gold_loans = gold_loans.drop('Occupation')

    # Check if any nulls exist and show rows with nulls
    has_nulls = gold_loans.na.drop().count() != gold_loans.count()
    print(f"Any nulls in dataframe: {has_nulls}")
    
    if has_nulls:
        # Show rows that contain any null values
        null_rows = gold_loans.filter(
            reduce(lambda x, y: x | y, [col(c).isNull() for c in gold_loans.columns])
        )
        print(f"\nRows with nulls (showing first 10):")
        null_rows.show(10)
        
        # Or show specific columns that have nulls
        print("\nColumns with null counts:")
        for column in gold_loans.columns:
            null_count = gold_loans.filter(col(column).isNull()).count()
            if null_count > 0:
                print(f"  - {column}: {null_count} nulls")

        raise AssertionError(
            f"DataFrame contains null values! "
            f"Total rows: {gold_loans.count()}, "
            f"Rows with nulls: {gold_loans.count() - gold_loans.na.drop().count()}, "
            f"Columns with nulls: {null_columns}"
        )

    print(f'Final gold_loans row count: {gold_loans.count()}')
    
    filepath = GOLD_INTEGRATED_DIR + 'gold_loans.parquet'
    gold_loans.write.mode('overwrite').parquet(filepath)
    print(f'saved to: {filepath}')
    
    return gold_loans