from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


model = VGG16(weights="imagenet", include_top=False)
model2 = ResNet50(weights="imagenet", include_top=False)


def image_to_array(image_list):
    """converts the image to array

    Parameters
    ----------
    image_list : [list]
        [a list that contains images]

    Returns
    -------
    [list]
        [returns a list of images that have been converted to array.]
    """
    array_image = []
    for image in images:
        array_image.append(img_to_array(image))
    return array_image


def process_image(image_array):
    """Processes images for the modelling part.

    Parameters
    ----------
    image_array : [list]
        [list of images that have been converted into array.]

    Returns
    -------
    [list]
        [returns a list of processed images.]
    """

    processed_image = []
    for item in image_array:
        processed_image.append(np.expand_dims(item, axis=0))

    for index in range(len(processed_image)):
        processed_image[index] = preprocess_input(processed_image[index])

    return processed_image


def extract_information_vg(image_array):
    """applying vgg16 model

    Parameters
    ----------
    image_array : [list]
        [a list of processed images that are numpy array]

    Returns
    -------
    [list]
        [a list of modelled images]
    """
    features = []
    for index in range(len(image_array)):
        features.append(model.predict(image_array[index]))
    return features


def extract_information_res(image_array):

    features = []
    for index in range(len(image_array)):
        features.append(model2.predict(image_array[index]))
    return features


def calculate_similarity(features):
    """calculates the cosine similarity between two matrices.

    Parameters
    ----------
    features : [list]
        [a list of modelled images]

    Returns
    -------
    [numpy]
        [returns the cosine score of the two images.]
    """

    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    similarity_score = cosine_loss(features[0], features[1]).numpy()

    return similarity_score


def plot_picture(image_array):
    """plots the images that have been converted to arrays.

    Parameters
    ----------
    image_array : [list]
        [a list of images that have been converted to arrays.]
    """
    for image in image_array:
        plt.imshow(np.uint8(image))
        plt.show()
