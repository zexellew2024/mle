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

from utils.constants import BRONZE_FEAT_DIR, FEATURE_FILENAMES


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/lms_loan_daily.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())
    
    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

def process_bronze_table_features(snapshot_date_str, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    if not os.path.exists(BRONZE_FEAT_DIR):
        os.makedirs(BRONZE_FEAT_DIR)

    return_dict = {}
    for feat_file_name in FEATURE_FILENAMES:
        csv_file_path = f"data/{feat_file_name}.csv"
    
        df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
        print()
        print(snapshot_date_str + '\trow count:', df.count()) 
        
        partition_name = f"bronze_{feat_file_name}" + snapshot_date_str.replace('-','_') + '.csv'
        filepath = BRONZE_FEAT_DIR + partition_name
        df.toPandas().to_csv(filepath, index=False)
        print('saved to:', filepath)

        return_dict[feat_file_name] = df

    return return_dict
