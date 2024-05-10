# Import necessary libraries
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import BinaryType, DoubleType, ArrayType, IntegerType, FloatType, StructField, StructType
import numpy as np
from PIL import Image
import io

# Initialize a SparkSession
spark = SparkSession.builder \
    .appName("CIFAR-10 Image Processing with PySpark") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memoryOverhead", "512m") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Function to unpickle the CIFAR-10 dataset files
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Function to load a CIFAR-10 batch file into a PySpark DataFrame
def load_cifar10_batch(file):
    """
    Loads a CIFAR-10 batch file and returns a list of tuples containing image data and labels.
    
    Args:
    - file (str): Path to the CIFAR-10 batch file.
    
    Returns:
    - list: A list of tuples, where each tuple contains (image_data, label).
    """
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    images_and_labels = []
    
    for i in range(len(data)):
        # Each row of data is an image
        image_array = data[i]
        # Reshape the array into 3x32x32 and transpose to 32x32x3
        image_array_reshaped = image_array.reshape(3, 32, 32).transpose(1, 2, 0)
        # Convert the image array to an image
        image = Image.fromarray(image_array_reshaped)
        # Convert the image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        # Append the image bytes and label to the list
        images_and_labels.append((image_bytes, labels[i]))
    
    return images_and_labels

# Paths to the CIFAR-10 batch files
batch_files = [
    "cifar-10-batches-py/data_batch_1",
    "cifar-10-batches-py/data_batch_2",
    "cifar-10-batches-py/data_batch_3",
    "cifar-10-batches-py/data_batch_4",
    "cifar-10-batches-py/data_batch_5",
    "cifar-10-batches-py/test_batch"
]

image_data_schema = StructType([
    StructField("image_data", BinaryType(), False),
    StructField("label", IntegerType(), False)
])

# all_images_and_labels = [image_label for file in batch_files for image_label in load_cifar10_batch(file)]
# rdd = spark.sparkContext.parallelize(all_images_and_labels)
# df = spark.createDataFrame(rdd, image_data_schema)

def create_dataframe_from_batch(file):
    images_and_labels = load_cifar10_batch(file)
    rdd = spark.sparkContext.parallelize(images_and_labels)
    return spark.createDataFrame(rdd, image_data_schema)

df = None
for batch_file in batch_files:
    batch_df = create_dataframe_from_batch(batch_file)
    if df is None:
        df = batch_df
    else:
        df.union(batch_df)

# Define a UDF that converts image byte data into a dense vector
def convert_bytes_to_vector(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    array =  np.array(image).flatten().astype(float) / 255.0
    return Vectors.dense(array)

convert_udf = udf(convert_bytes_to_vector, VectorUDT())

df = df.withColumn("features", convert_udf("image_data"))

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10)

pipeline = Pipeline(stages=[rf])

model = pipeline.fit(train_df)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(model.transform(test_df))
print(f"Accuracy: {accuracy}")

# Stop the SparkSession
spark.stop()
