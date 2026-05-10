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

from pyspark.sql.functions import col, when, trim, regexp_replace, regexp_extract, explode, split
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.constants import BRONZE_FEAT_DIR, SILVER_FEAT_DIR, FEATURE_FILENAMES


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_table_features(snapshot_date_str, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    if not os.path.exists(SILVER_FEAT_DIR):
        os.makedirs(SILVER_FEAT_DIR)

    return_dict = {}
    for feat_file_name in FEATURE_FILENAMES:
        partition_name = f"bronze_{feat_file_name}" + snapshot_date_str.replace('-','_') + '.csv'
        filepath = BRONZE_FEAT_DIR + partition_name
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        print('loaded from:', filepath, 'row count:', df.count())
        
        # processing
        df = clean(feat_file_name, df)
        
        partition_name = f"silver_{feat_file_name}" + snapshot_date_str.replace('-','_') + '.parquet'
        filepath = SILVER_FEAT_DIR + partition_name
        df.write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath)

        return_dict[feat_file_name] = df

    return return_dict

def clean(feat_file_name, df):
    match feat_file_name:
    
        case 'feature_clickstream':
            column_type_map = {f'fe_{i}': IntegerType() for i in range(1, 21)}
            for column, new_type in column_type_map.items():
                df = df.withColumn(column, col(column).cast(new_type))
                
        case 'features_attributes':
            df = df.drop('Name') # Not useful for ML
            df = df.drop('SSN')  # Should not be used - privacy

            # For age, do some validation. Remove _, then only keep those within 1 to 100. None for the rest.
            df = clean_numeric(df, 'Age', IntegerType(), 1, 100)

            # Also clean occupation
            df = df.withColumn('Occupation',
                when(
                    ~col('Occupation').rlike(r'_{2,}'),  # Does NOT contain 2+ underscores
                    trim(col('Occupation'))               # Keep trimmed value
                ).otherwise(None)                         # Set to null if has multiple underscores
            )

            distinct_occupations = df.filter(col('Occupation').isNotNull()) \
                                 .select('Occupation') \
                                 .distinct() \
                                 .rdd.flatMap(lambda x: x) \
                                 .collect()
        
            # Create a column for each occupation (1 if matches, 0 otherwise)
            for occupation in distinct_occupations:
                # Create a valid column name (replace spaces/special chars with underscore)
                col_name = f"is_{occupation.replace(' ', '_').replace('-', '_')}"
                df = df.withColumn(col_name, 
                                   when(col('Occupation') == occupation, 1).otherwise(0).cast(IntegerType()))
            
        case 'features_financials':

            df = clean_numeric(df, 'Annual_Income', FloatType(), 0)
            df = clean_numeric(df, 'Monthly_Inhand_Salary', FloatType(), 0)

            df = clean_numeric(df, 'Num_Bank_Accounts', IntegerType(), 0, 100)
            df = clean_numeric(df, 'Num_Credit_Card', IntegerType(), 0, 100)
            df = clean_numeric(df, 'Interest_Rate', IntegerType(), 0, 100)
            df = clean_numeric(df, 'Num_of_Loan', IntegerType(), 0, 100)

            df = clean_numeric(df, 'Delay_from_due_date', IntegerType(), 0)
            df = clean_numeric(df, 'Num_of_Delayed_Payment', IntegerType(), 0)
            df = clean_numeric(df, 'Changed_Credit_Limit', FloatType())
            df = clean_numeric(df, 'Num_Credit_Inquiries', IntegerType(), 0)

            df = clean_numeric(df, 'Outstanding_Debt', FloatType(), 0)
            df = clean_numeric(df, 'Credit_Utilization_Ratio', FloatType(), 0)
            df = clean_numeric(df, 'Total_EMI_per_month', FloatType(), 0)
            
            df = clean_numeric(df, 'Amount_invested_monthly', FloatType(), 0)
            df = clean_numeric(df, 'Monthly_Balance', FloatType())
                        
            # Handle the loan columns, set binary flags
            loan_types = df \
                .withColumn('Type_of_Loan', regexp_replace(col('Type_of_Loan'), '\\band\\b', ',')) \
                .withColumn('loan', explode(split(col('Type_of_Loan'), ','))) \
                .select('loan') \
                .distinct() \
                .collect()
            
            # Create binary flags
            for row in loan_types:
                loan = row.loan.strip()
                if loan and loan != '':  # Skip empty
                    col_name = f'has_{loan.lower().replace(" ", "_").replace("-", "_")}'
                    df = df.withColumn(col_name, 
                        when(col('Type_of_Loan').contains(loan), 1).otherwise(0))
            
            # Drop original column
            df = df.drop('Type_of_Loan')

            # Handle the Credit Mix
            df = df.withColumn('Credit_Mix_Label',
                when(col('Credit_Mix') == 'Good', 2)
                .when(col('Credit_Mix') == 'Standard', 1)
                .when(col('Credit_Mix') == 'Bad', 0)
                .otherwise(None)  # '_' and anything else becomes null
            )
        
            # Drop original column 
            df = df.drop('Credit_Mix')

            # Handle Credit History Age
            df = df.withColumn('Credit_History_Months',
                (regexp_extract(col('Credit_History_Age'), r'(\d+) Years', 1).cast('int') * 12 +
                 regexp_extract(col('Credit_History_Age'), r'(\d+) Months', 1).cast('int'))
            )
            df = df.drop('Credit_History_Age')

            # Handle Payment_of_Min_Amount. Assume NM (not mentioned), means 0.
            df = df.withColumn('Payment_of_Min_Amount', 
                when(col('Payment_of_Min_Amount') == 'Yes', 1).otherwise(0)
            )

            # For payment behaviour, split into 2 features
            df = df.withColumn('Spending_Level',
                when(col('Payment_Behaviour').rlike('^High_spent'), 1)
                .when(col('Payment_Behaviour').rlike('^Low_spent'), 0)
                .otherwise(None)
            )

            df = df.withColumn('Payment_Value',
                when(col('Payment_Behaviour').rlike('High_value'), 2)
                .when(col('Payment_Behaviour').rlike('Medium_value'), 1)
                .when(col('Payment_Behaviour').rlike('Small_value'), 0)
                .otherwise(None)
            )
            
            df = df.drop('Payment_Behaviour')

    return df

def clean_numeric(df, col_name, targetType, min_val = float('-inf'), max_val = float('inf')):
    """
    Many numeric cols have underscores. This function removes them.
    If specified, also ensures the value lies between min and max val.
    Also nullifies values with leading/trailing underscores (e.g., __10000__)
    """
    # First, check for leading/trailing underscores
    df = df.withColumn(col_name,
        when(
            col(col_name).rlike(r'^_.*|.*_$'),  # Starts or ends with underscore
            None  
        ).otherwise(col(col_name))
    )
    
    # Then remove internal underscores and validate
    df = df.withColumn(col_name, 
        when(
            col(col_name).isNotNull(),
            regexp_replace(col(col_name), '_', '')  # Remove remaining underscores
        ).otherwise(None)
    )
    
    df = df.withColumn(col_name, 
        when(
            col(col_name).rlike(r'^-?\d+(?:\.\d+)?$') &  # Allows decimals
            col(col_name).cast(targetType).between(min_val, max_val), 
            col(col_name).cast(targetType)
        ).otherwise(None)
    )
    return df
