from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("Test App").getOrCreate()
print(spark.version)
