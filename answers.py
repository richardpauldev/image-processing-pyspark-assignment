# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, desc
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

# TASK 1

# Function to resize images to 64x64 pixels
def resize_image(image_bytes):
    """
    Resize the given image to 64x64 pixels.
    
    Args:
    - image_bytes (bytes): The original image in byte format.
    
    Returns:
    - bytes: The resized image in byte format.
    """
    image = Image.open(io.BytesIO(image_bytes))
    resized_image = image.resize((64, 64), Image.ANTIALIAS)
    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# Register the UDF
resize_image_udf = udf(resize_image, BinaryType())

# Resize images in the DataFrame
df = df.withColumn("image_data", resize_image_udf(col("image_data")))

# TASK 2

def normalize_image(image_bytes):
    """
    Normalize the pixel values of an image so that it has a mean of 0 and a standard deviation of 1.
    
    Args:
    - image_bytes (bytes): The image in byte format.
    
    Returns:
    - bytes: The normalized image in byte format.
    """
    image = Image.open(io.BytesIO(image_bytes))
    np_image = np.array(image).astype(float)
    mean = np_image.mean()
    std = np_image.std()
    normalized_image = (np_image - mean) / std
    normalized_image = np.clip(normalized_image, -1, 1)  # Clip values to prevent overflow in uint8
    normalized_image = (255 * (normalized_image - normalized_image.min()) / (normalized_image.max() - normalized_image.min())).astype(np.uint8)
    img_byte_arr = io.BytesIO()
    Image.fromarray(normalized_image).save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# Register the UDF
normalize_image_udf = udf(normalize_image, BinaryType())

# Resize images in the DataFrame
df = df.withColumn("image_data", normalize_image_udf(col("image_data")))

# TASK 3
label_count = df.groupBy("label").count().orderBy("count")

most_common = label_count.orderBy(desc("count")).first()
least_common = label_count.orderBy("count").first()

label_count.show()
print(f"The most common category is label {most_common['label']} with {most_common['count']} images.")
print(f"The least common category is label {least_common['label']} with {least_common['count']} images.")

# Show the schema of the DataFrame
df.printSchema()
df.show(5)

# Stop the SparkSession
spark.stop()
