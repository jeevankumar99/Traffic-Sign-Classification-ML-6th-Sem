"""trains a model using the many a corpus of traffic pictures to recognise the signs"""

import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    print(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []
    
    # iterate through directories ( 0 to 42)
    for i in range(0, NUM_CATEGORIES):
        image_path = None
        image_path = os.listdir(data_dir + os.sep + str(i))
        
        # iterate through images in each directory
        for image in image_path:
            img = cv2.imread(data_dir + os.sep + str(i) + os.sep + image, 1)
            images.append(cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)))
            labels.append(i)
    
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([

        # Add 32 filters using 4 by 4 kernel and pool at 2 by 2.
        tf.keras.layers.Conv2D(32, (4, 4), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        
        # Add 64 filters using 4 by 4 kernal and pool by 2 by 2.
        tf.keras.layers.Conv2D(64, (4, 4), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        
        # Flatten the images after convolution and pooling
        tf.keras.layers.Flatten(),
        
        # Add 144 hidden layers and set dropout to 0.5 to avoid dependency on particular nodes.
        tf.keras.layers.Dense(144, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add 43 output layers as 43 different signs are present.
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model


if __name__ == "__main__":
    main()
