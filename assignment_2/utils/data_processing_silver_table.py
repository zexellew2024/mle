import os
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, trim, regexp_replace, regexp_extract, log1p
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.constants import BRONZE_FEAT_DIR, SILVER_FEAT_DIR, FEATURE_FILENAMES


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

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

    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_table_features(snapshot_date_str, spark):
    if not os.path.exists(SILVER_FEAT_DIR):
        os.makedirs(SILVER_FEAT_DIR)

    return_dict = {}
    for feat_file_name in FEATURE_FILENAMES:
        partition_name = f"bronze_{feat_file_name}" + snapshot_date_str.replace('-', '_') + '.csv'
        filepath = BRONZE_FEAT_DIR + partition_name
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        print('loaded from:', filepath, 'row count:', df.count())

        df = clean(feat_file_name, df)

        partition_name = f"silver_{feat_file_name}" + snapshot_date_str.replace('-', '_') + '.parquet'
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
            df = df.drop('Name')  # Not useful for ML
            df = df.drop('SSN')   # Should not be used - privacy

            # For age, do some validation. Remove _, then only keep those within 1 to 100. None for the rest.
            df = clean_numeric(df, 'Age', IntegerType(), 1, 100)

            # Also clean occupation
            df = df.withColumn('Occupation',
                when(
                    ~col('Occupation').rlike(r'_{2,}'),  # Does NOT contain 2+ underscores
                    trim(col('Occupation'))
                ).otherwise(None)
            )

            # Bin age into groups, keeping null Age as null Age_group
            df = df.withColumn('Age_group',
                when(col('Age').isNull(), None)
                .when(col('Age') < 25, '<25')
                .when(col('Age') < 40, '25-39')
                .when(col('Age') < 60, '40-59')
                .otherwise('60+')
            )

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

            # Count distinct loan types instead of per-type dummy columns (stable across partitions)
            df = df.withColumn('Loan_Types_Array',
                when(col('Type_of_Loan').isNotNull(),
                     F.array_distinct(F.expr("transform(split(Type_of_Loan, ', |, and '), x -> lower(trim(x)))")))
            )
            df = df.withColumn('num_loan_types',
                when(col('Loan_Types_Array').isNotNull(), F.size(col('Loan_Types_Array'))).otherwise(0))
            df = df.drop('Type_of_Loan', 'Loan_Types_Array')

            # Handle the Credit Mix
            df = df.withColumn('Credit_Mix_Label',
                when(col('Credit_Mix') == 'Good', 2)
                .when(col('Credit_Mix') == 'Standard', 1)
                .when(col('Credit_Mix') == 'Bad', 0)
                .otherwise(None)  # '_' and anything else becomes null
            )
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

            # Ratio features computed here while raw leakage columns are still present
            df = df.withColumn('log_Annual_Income', log1p(col('Annual_Income')))
            df = df.withColumn('debt_to_income', col('Outstanding_Debt') / (col('Annual_Income') + 1))
            df = df.withColumn('emi_to_salary', col('Total_EMI_per_month') / (col('Monthly_Inhand_Salary') + 1))
            df = df.withColumn('investment_rate', col('Amount_invested_monthly') / (col('Monthly_Inhand_Salary') + 1))
            df = df.withColumn('balance_to_debt', (col('Monthly_Balance') + 1) / (col('Outstanding_Debt') + 1))
            df = df.withColumn('inq_per_loan', col('Num_Credit_Inquiries') / (col('Num_of_Loan') + 1))
            df = df.withColumn('has_credit_limit_change', (col('Changed_Credit_Limit') != 0).cast(IntegerType()))

    return df


def clean_numeric(df, col_name, targetType, min_val=float('-inf'), max_val=float('inf')):
    """
    Many numeric cols have underscores. This function removes them.
    If specified, also ensures the value lies between min and max val.
    Also nullifies values with leading/trailing underscores (e.g., __10000__)
    """
    df = df.withColumn(col_name,
        when(
            col(col_name).rlike(r'^_.*|.*_$'),
            None
        ).otherwise(col(col_name))
    )

    df = df.withColumn(col_name,
        when(
            col(col_name).isNotNull(),
            regexp_replace(col(col_name), '_', '')
        ).otherwise(None)
    )

    df = df.withColumn(col_name,
        when(
            col(col_name).rlike(r'^-?\d+(?:\.\d+)?$') &
            col(col_name).cast(targetType).between(min_val, max_val),
            col(col_name).cast(targetType)
        ).otherwise(None)
    )
    return df
