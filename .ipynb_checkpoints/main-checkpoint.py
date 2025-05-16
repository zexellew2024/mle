from pyspark.sql import SparkSession 
import os 

def create_spark_session():
    return SparkSession.builder.appName("LoanFeaturePipeline").getOrCreate()

def main():
    print("Pipeline starts: ")

def init_datamart():
    layers = ['bronze', 'silver', 'gold'] 
    for layer in layers:
        path = os.path.join("datamart", layer)
        os.makedirs(path, exist_ok = True)

def ingest_raw_to_bronze(spark):
    data_sources = {
        "clickstream": "data/feature_clickstream.csv", 
        "attributes": "data/features_attributes.csv", 
        "financials": "data/features_financials.csv", 
        "loans": "data/lms_loan_daily.csv"
    }

    for name, path in data_sources.items():
        print(f" Ingesting {name} from {path}")
        df = spark.read.csv(path, header=True, inferSchema=True)
        df.write.mode("overwrite").parquet(f"datamart/bronze/{name}")
        print(f"Saved bronze/{name}")

if __name__ == "__main__": 
    main()
    init_datamart()
    spark = create_spark_session()
    ingest_raw_to_bronze(spark)
    print("Created datamart folders!")
    