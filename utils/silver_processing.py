from pyspark.sql.functions import col, regexp_replace, when, split, size, array_contains, expr, ceil, datediff
from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType

#CLEAN FINANCIALS
def clean_financials_table(spark):
    print("Cleaning financials...") 

    #Load bronze 
    df = spark.read.parquet("datamart/bronze/financials")

    #Replace messsy entries with None
    df = df.replace(["_", "NA", "na", "N/A"], None)

    #Remove formatting noise from numeric cols: "__52424"
    df = df.withColumn("Annual_Income", regexp_replace(col("Annual_Income"), "[^0-9.]", ""))
    df = df.withColumn("Num_of_Loan", regexp_replace(col("Num_of_Loan"), "[^0-9]", ""))
    df = df.withColumn("Num_of_Delayed_Payment", regexp_replace(col("Num_of_Delayed_Payment"), "[^0-9]", ""))
    df = df.withColumn("Amount_invested_monthly", regexp_replace(col("Amount_invested_monthly"), "[^0-9.]", ""))
    
    # Define schema enforcement using a dictionary
    cast_type_map = {
        "Annual_Income": DoubleType(),
        "Monthly_Balance": DoubleType(),
        "Outstanding_Debt": DoubleType(),
        "Amount_invested_monthly": DoubleType(),
        "Changed_Credit_Limit": DoubleType(),
        "Total_EMI_per_month": DoubleType(),
        "Credit_Utilization_Ratio": DoubleType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Num_Credit_Inquiries": IntegerType()
    }


    for column, dtype in cast_type_map.items():
        df = df.withColumn(column, col(column).cast(dtype))

    #Clean Payment behaviour 
    valid_payment_behaviours = [
        "High_spent_Large_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Small_value_payments"
    ]
    df = df.withColumn(
        "Payment_Behaviour",
        when(col("Payment_Behaviour").isin(valid_payment_behaviours), col("Payment_Behaviour"))
        .otherwise(None)
    )

    # Preprocess Type_of_Loan
    df = df.withColumn("Loan_Types_Array", split(col("Type_of_Loan"), ", |, and "))
    df = df.withColumn("Loan_Types_Array", expr("transform(Loan_Types_Array, x -> lower(trim(x)))"))

    # Remove rows with null key identifiers
    df = df.filter(col("Customer_ID").isNotNull() & col("snapshot_date").isNotNull())

    # Write to silver
    df.write.mode("overwrite").parquet("datamart/silver/financials_clean")

    print("✅Saved cleaned customer financials data to silver/financials_clean")

    return df

#CLEAN ATTRIBUTES

def clean_attributes_table(spark): 
    print("Cleaning customer attributes...")
    #Load bronze 
    df = spark.read.parquet("datamart/bronze/attributes")

    #Replace dirty entries with none
    df = df.replace(["_", "NA", "na", "N/A", "_______"], None)

    #Clean Age: remove non-numeric, cast to int, filter out wrongly formatted values 
    df = df.withColumn("Age", regexp_replace(col("Age"), "[^0-9]", ""))
    df = df.withColumn("Age", when(
    (col("Age").cast("int") > 0) & (col("Age").cast("int") < 100),
    col("Age").cast(IntegerType())
).otherwise(None))
    #Clean SSN: 
    df = df.withColumn("SSN", when(col("SSN").rlike("^\\d{3}-\\d{2}-\\d{4}$"), col("SSN")).otherwise(None))

    #Drop rows with missing primary keys/timestamps:
    df = df.filter(col("Customer_ID").isNotNull() & col("snapshot_date").isNotNull())

    df.write.mode("overwrite").parquet("datamart/silver/attributes_clean")

    print("✅ Saved cleaned customer attributes data to silver/attributes_clean!")

    return df 

#CLEAN CLICKSTREAM

def clean_clickstream_table(spark):
    print("Cleaning clickstream features...")
    df = spark.read.parquet("datamart/bronze/clickstream")
    df = df.replace(["_", "NA", "na", "N/A"], None)

    # Explicitly cast fe_1 to fe_20 to IntegerType
    for i in range(1, 21):
        colname = f"fe_{i}"
        df = df.withColumn(colname, col(colname).cast(IntegerType()))

    df = df.filter(col("Customer_ID").isNotNull() & col("snapshot_date").isNotNull())
    df.write.partitionBy("snapshot_date").mode("overwrite").parquet("datamart/silver/clickstream_clean")
    print("✅ Saved cleaned customer clickstream to silver/clickstream_clean")
    return df


#CLEAN LOAN

def clean_loans_table(spark):
    print("Cleaning loan data...")

    df = spark.read.parquet("datamart/bronze/loans") 

    schema_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": DoubleType(),
        "due_amt": DoubleType(),
        "paid_amt": DoubleType(),
        "overdue_amt": DoubleType(),
        "balance": DoubleType(),
        "snapshot_date": DateType()
    }

    for colname, dtype in schema_map.items():
        df = df.withColumn(colname, col(colname).cast(dtype))

    df = df.withColumn("mob", col("installment_num"))
    df = df.withColumn("installments_missed", ceil(col("overdue_amt") / col("due_amt")))
    df = df.withColumn("first_missed_date", when(col("installments_missed") > 0, expr("add_months(snapshot_date, -installments_missed)")).otherwise(None))
    df = df.withColumn("dpd", when(col("overdue_amt") > 0.0, datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))
    
    df = df.filter(col("Customer_ID").isNotNull() & col("snapshot_date").isNotNull())
    df.write.partitionBy("snapshot_date").mode("overwrite").parquet("datamart/silver/loans_clean")
    print("✅ Saved cleaned loans to silver/loans_clean")
    return df

                       
    

    
    
    

