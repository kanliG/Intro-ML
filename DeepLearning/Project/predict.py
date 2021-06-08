# built and trained a deep neural network on the flower data set

# Import TensorFlow
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

#  Ingore error TensorFlow
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import time
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import os

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from PIL import Image


# Load oxford_flowers102 dataset
def load_dataset():
    # Load the dataset with TensorFlow Datasets.
    dataset_f, dataset_info_f = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
    # Create a training set, a validation set and a test set.
    training_set_f, validation_set_f, test_set_f = dataset_f['train'], dataset_f['validation'], dataset_f['test']
    return training_set_f, validation_set_f, test_set_f, dataset_f, dataset_info_f

# Normalize
def normalize_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size)
    image /= 255
    return image, label


# Create the process_image function
def process_image(image):
    image=tf.convert_to_tensor(image,tf.float32)
    image=tf.image.resize(image, (224, 224))
    image/=255
    return image


# Load image test from specific folder
def load_images_path(folder):
    images_path = []
    for filename in os.listdir(folder):
        images_path.append(os.path.join(folder,filename))
    return images_path


# Process_image function we have provided 4 images in the ./test_images/ folder
def process_test_image():
    folder = './test_images/'
    filenames = load_images_path(folder)

    for image_path in filenames:
        im = Image.open(image_path)
        test_image = np.asarray(im)
        processed_test_image = process_image(test_image)
    return processed_test_image


# Create the predict function
def predict(image_path=None, model=None, top_k=None):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image=np.expand_dims(processed_test_image,0)
    probs = model.predict(processed_test_image)
    return tf.nn.top_k(probs, k=top_k)


def filtered(classes):
    return [class_names.get(str(key)) if key else "Placeholder" for key in classes.numpy().squeeze().tolist()]


# Display the predict result
def prediction_result(filename):
    probs, classes = predict(image_path=filename, model=model, top_k=5)
    print(f"\u2022 File: {filename} \n\u2022 Probability: {probs[0]}\n\u2022 Classes: {classes}")

    im = Image.open(filename)
    test_image = np.asarray(im)
    axis_label=filtered(classes)
    fig,(ax1,ax2)=plt.subplots(figsize=(10,3),ncols=2)
    ax1.imshow(test_image,cmap=plt.cm.binary)
    ax2.set_title('Class Probability')
    ax2.barh(np.array(axis_label),probs[0])

# Main function
if __name__ == '__main__':
    # Dataset
    training_set, validation_set, test_set, dataset, dataset_info = load_dataset()

    # Hyper parameters
    batch_size = 34
    image_size = (224, 224)
    total_training_samples = num_validation = dataset_info.splits['validation'].num_examples


    # Create the training, validation and test batches
    training_batches = training_set.shuffle(total_training_samples // 2).map(normalize_image).batch(
        batch_size).prefetch(1)
    validation_batches = validation_set.map(normalize_image).batch(batch_size).prefetch(1)
    test_batches = test_set.map(normalize_image).batch(batch_size).prefetch(1)

    # Load the class names
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)

    # Load the model
    model = tf.keras.models.load_model(str('./' + 'saved_model'), custom_objects={'KerasLayer': hub.KerasLayer})

    # Load the model
    folder = './test_images/'
    filenames = load_images_path(folder)

    for image_path in filenames:
        prediction_result(image_path)



