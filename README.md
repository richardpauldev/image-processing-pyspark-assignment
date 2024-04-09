# Building a Data Processing Pipeline with PySpark for Image Processing

## Assignment Instructions

### Objectives
 - Set up a PySpark environment.
 - Load a dataset of images into PySpark DataFrames.
 - Perform basic preprocessing on image data to prepare for further analysis.

### Requirements
 - Python 3.x
 - PySpark
 - An image dataset (provided)

### Task 1: Image Resizing

    Implement a function to resize images to 64x64 pixels.
    Apply this function to the image_data column in the DataFrame.
    Ensure the resized images are still in a format that can be processed by PySpark.

### Task 2: Image Normalization

    Implement a function to normalize the pixel values of images.
    The function should adjust the pixel values so that each image has a mean of 0 and a standard deviation of 1.
    Apply this normalization function to the resized images.

### Task 3: Label Distribution Analysis

    Using the label column in the DataFrame, count the number of images for each label.
    Identify the most and least common categories in the CIFAR-10 dataset.
    Provide a brief analysis of the label distribution. Discuss any potential impacts this distribution might have on training a machine learning model.

### Submission Guidelines
 - Submit a single Python file (.py) containg you PySpark code.
 - Ensure your code is well-commented to explain your logic and choices.
 - Include a brief report explaining your approach, any challenges you faced, and how you overcame them.

 ### Resources

 1. [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/index.html)
 2. [Spark Quick Start Guide](https://spark.apache.org/docs/latest/quick-start.html)
 3. [Pillow Documentation](https://pillow.readthedocs.io/en/stable/)
