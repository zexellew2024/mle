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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.constants import GOLD_FEAT_DIR, SILVER_FEAT_DIR, FEATURE_FILENAMES


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