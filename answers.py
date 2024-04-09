from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType
from PIL import Image
import numpy as np
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

# UDF to resize images
def resize_image(image_data, size=(224, 224)):
    """
    Resizes an image to the specified size.
    
    Args:
    - image_data (bytearray): The image data in bytes.
    - size (tuple): The target size (width, height).
    
    Returns:
    - bytearray: The resized image data.
    """
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))
    # Resize the image
    resized_image = image.resize(size)
    # Convert the resized image to bytes
    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# UDF to normalize images
def normalize_image(image_data):
    """
    Normalizes pixel values in the image to be between 0 and 1.
    
    Args:
    - image_data (bytearray): The image data in bytes.
    
    Returns:
    - bytearray: The normalized image data.
    """
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize the image array
    normalized_image_array = image_array.astype(np.float32) / 255.0
    # Convert the numpy array back to an image
    normalized_image = Image.fromarray((normalized_image_array * 255).astype(np.uint8))
    # Convert the normalized image to bytes
    img_byte_arr = io.BytesIO()
    normalized_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# Register the UDFs
resize_image_udf = udf(resize_image, BinaryType())
normalize_image_udf = udf(normalize_image, BinaryType())

# Assuming df is your DataFrame containing the image data
# Apply the resize and normalize UDFs
df = df.withColumn("resized_image", resize_image_udf("image_data"))
df = df.withColumn("normalized_image", normalize_image_udf("resized_image"))

# Show the updated DataFrame schema
df.printSchema()

# Note: The actual DNN model training and prediction would typically occur outside of PySpark,
# using a deep learning library such as TensorFlow or PyTorch. This preprocessing prepares the
# image data for such use.

# Stop the SparkSession
spark.stop()
