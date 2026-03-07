from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Test Spark") \
    .getOrCreate()

data = [("Anantha",25),("Rahul",30)]

df = spark.createDataFrame(data,["Name","Age"])

df.show()

spark.stop()
