# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import BinaryType, DoubleType, ArrayType, IntegerType
import numpy as np
from PIL import Image
import io

# Initialize a SparkSession
spark = SparkSession.builder \
    .appName("CIFAR-10 Image Processing with PySpark") \
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

# Example path to a CIFAR-10 batch file (update this path to your CIFAR-10 batch file location)
cifar10_batch_file = "cifar-10-batches-py/data_batch_1"

# Load a CIFAR-10 batch
images_and_labels = load_cifar10_batch(cifar10_batch_file)

# Convert the list of tuples into a DataFrame
schema = ["image_data", "label"]
df = spark.createDataFrame(images_and_labels, schema=schema)

# Show the schema of the DataFrame
df.printSchema()
df.show(5)

# TODO: Implement tasks 1,2,3 as described in the README

# Stop the SparkSession
spark.stop()
