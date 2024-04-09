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

# Completed Code: 
# Task 1: Image Resizing
def resize_image(image_data, size=(64, 64)):
    """
    Resizes an image to the specified size.
    
    Args:
    - image_data (binary): The original image data.
    - size (tuple): The target size (width, height).
    
    Returns:
    - binary: The resized image data.
    """
    # Convert binary data to PIL Image
    image = Image.open(io.BytesIO(image_data))
    # Resize image
    resized_image = image.resize(size)
    # Convert back to binary
    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

resize_udf = udf(resize_image, BinaryType())
df_resized = images_and_labels.withColumn("resized_image_data", resize_udf("image_data"))

# Task 2: Image Normalization
def normalize_image(image_data):
    """
    Normalizes an image so that its pixel values have mean 0 and std 1.
    
    Args:
    - image_data (binary): The image data.
    
    Returns:
    - binary: The normalized image data.
    """
    # Convert binary data to PIL Image
    image = Image.open(io.BytesIO(image_data))
    # Convert image to numpy array
    img_array = np.array(image)
    # Normalize
    normalized_img_array = (img_array - np.mean(img_array)) / np.std(img_array)
    # Ensure the normalized array still has values between 0 and 255
    normalized_img_array = ((normalized_img_array - normalized_img_array.min()) / (normalized_img_array.max() - normalized_img_array.min()) * 255).astype(np.uint8)
    # Convert numpy array back to PIL Image
    normalized_image = Image.fromarray(normalized_img_array)
    # Convert back to binary
    img_byte_arr = io.BytesIO()
    normalized_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

normalize_udf = udf(normalize_image, BinaryType())
df_normalized = df_resized.withColumn("normalized_image_data", normalize_udf("resized_image_data"))

# Task 3: Label Distribution Analysis
label_distribution = df_normalized.groupBy("label").count().orderBy("count", ascending=False)
label_distribution.show()

# Stop the SparkSession
spark.stop()
