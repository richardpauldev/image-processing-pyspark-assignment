# Import necessary libraries
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import BinaryType, DoubleType, ArrayType, IntegerType, FloatType
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sparkdl import KerasImageFileTransformer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This tells TensorFlow to use only the CPU


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

def keras_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Paths to the CIFAR-10 batch files
batch_files = [
    "cifar-10-batches-py/data_batch_1",
    "cifar-10-batches-py/data_batch_2",
    "cifar-10-batches-py/data_batch_3",
    "cifar-10-batches-py/data_batch_4",
    "cifar-10-batches-py/data_batch_5",
    "cifar-10-batches-py/test_batch"
]

def PIL_image_loader(path):
    """ Load an image file into a 3D Numpy array. """
    img = Image.open(path)
    img = img.resize((32, 32))
    arr = np.array(img).astype(np.float32)
    # Normalize to [0, 1]
    arr /= 255.0
    return arr

# Process all batches into a single DataFrame
df = spark.createDataFrame([(load_cifar10_batch(f)) for f in batch_files], ["image_data", "label"])

transformer = KerasImageFileTransformer(inputCol="image_data", outputCol="predictions", model=keras_model, imageLoader=PIL_image_loader, outputMode="vector")

df = transformer.transform(df)

df.select("predictions").show(5)


# Stop the SparkSession
spark.stop()
