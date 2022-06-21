# Takes in an image and runs it through the model

import tensorflow as tf
import numpy as np
import sys
import cv2

from traffic import IMG_HEIGHT, IMG_WIDTH

LABELS_TO_STRINGS = {
    0: "20 Speed Limit",
    1: "30 Speed Limit",
    2: "50 Speed Limit",
    3: "80 Speed Limit",
    4: "70 Speed Limit",
    5: "80 Max Speed Limit",
    6: "80 Min Speed Limit",
    8: "100 Speed Limit",
    9: "120 Speed Limit",
    10: "Cars on right lane",
    11: "School nearby",
    12: "Caution Sign",
    13: "Slow down Sign",
    14: "Stop Sign",
    15: "Empty sign",
    16: "Heavy Duty Vehicle",
    17: "No Entry",
    18: "Exclamation Sign",
    19: "Left turn sign",
    20: "Right turn sign",
    21: "Left Chicane Sign",
    22: "Speed bumps ahead",
    23: "Slippery road ahead",
    24: "Merge Lane sign",
    25: "Construction work ahead",
    26: "Traffic Lights ahead",
    27: "Pedestrians Crossing sign",
    28: "Children Crossing sign",
    29: "Cyclists ahead",
    30: "Snow ahead",
    31: "Wildlife ahead",
    32: "Road Closed Permanently",
    33: "Right turn Cyclists",
    34: "Left turn Cyclists",
    35: "Straight ahead Cyclists",
    36: "Right Lane turn Cyclists",
    37: "Left lane turn Cyclists",
    38: "Right Exit Expressway",
    39: "Left Exit Expressway",
    40: "Round About sign",
    41: "No LMV(cars) sign",
    42: "No cars and trucks sign"
}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test_images/image1.ppm"
    
    # Read and resize image
    img = cv2.imread(image_path, 1)
    resized_image = cv2.resize(img, (30, 30))
    print("shape:", resized_image.shape)
    
    # load model
    model = tf.keras.models.load_model("trainer.h5")
    
    # reshape np array and predict sample
    probs = model.predict(np.reshape(resized_image, (1, 30, 30, 3)))
    
    # pick labels with highest probability and print
    sorted_probs = probs.argsort()
    label = LABELS_TO_STRINGS[sorted_probs[0][-1]]
    print("\nPredicted Labels of image: ", label)
    print ("\n")

