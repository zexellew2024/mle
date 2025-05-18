from pyspark.sql import SparkSession 
import os 
from utils.bronze_processing import ingest_bronze_tables
from utils.silver_processing import clean_financials_table, clean_attributes_table, clean_clickstream_table, clean_loans_table 


def create_spark_session():
    spark = SparkSession.builder.appName("LoanFeaturePipeline").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR") 
    return spark 

def init_datamart():
    layers = ['bronze', 'silver', 'gold'] 
    for layer in layers:
        path = os.path.join("datamart", layer)
        os.makedirs(path, exist_ok = True)

if __name__ == "__main__": 
    spark = create_spark_session()
    init_datamart()
    ingest_bronze_tables(spark)
    print("🥉Bronze stages complete!") 

    clean_financials_table(spark)
    print("Silver stage (financials) complete!")

    clean_attributes_table(spark)
    print("Silver stage (attributes) complete!")

    clean_clickstream_table(spark)
    print("Silver stage (clickstream) complete!")

    clean_loans_table(spark)
    print("Silver stage (loans) complete!")

    print("🥈Silver stages complete!") 

    