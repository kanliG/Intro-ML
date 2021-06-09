# built and trained a deep neural network on the flower data set

# Import TensorFlow
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
tfds.disable_progress_bar()

#  Ingore error TensorFlow
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import tkinter as tk
from PIL import Image
from pathlib import Path
from tkinter import messagebox    
import sys

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# Load oxford_flowers102 dataset
def load_dataset():
    # Load the dataset with TensorFlow Datasets.
    dataset_f, dataset_info_f = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
    # Create a training set, a validation set and a test set.
    training_set_f, validation_set_f, test_set_f = dataset_f['train'], dataset_f['validation'], dataset_f['test']
    return training_set_f, validation_set_f, test_set_f, dataset_f, dataset_info_f

def duild_model(num_classes):
    # Load MobileNet pre-trained network (feature_vector, without classification layer as we will add our own)
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL, input_shape=(224, 224, 3))
    feature_extractor.trainable = False
    model = tf.keras.Sequential([feature_extractor,
                                 tf.keras.layers.Dense(num_classes, activation='softmax')])
    # Stop training when there is no improvement in the validation loss for 5 consecutive epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # Compile the model with optimizer, loss and metrics parameters
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    return model


# Train the model and validate
def train_model(model, EPOCHS):
    train_history = model.fit(training_batches,
                          epochs=EPOCHS,
                          validation_data=validation_batches)
    return model, train_history


# Plot the loss and accuracy values achieved during training for the training and validation set.   
def plot_progress(train_history):
    training_accuracy = train_history.history['accuracy']
    validation_accuracy = train_history.history['val_accuracy']
    training_loss = train_history.history['loss']
    validation_loss = train_history.history['val_loss']
    epochs_range = range(len(training_accuracy))
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, training_loss, label='Training Loss')
    plt.plot(epochs_range, validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()    
    plt.savefig('./prediction_result/valos.png')


# Print the loss and accuracy values achieved on the entire test set.
def evaluate(model):
    loss, accuracy = model.evaluate(test_batches)
    return loss, accuracy

# Check some predictions to see visually how the model is performing
def plot_prediction(test_batches):
    for image_batch, label_batch in test_batches.take(1):
        ps = model.predict(image_batch)
        images = image_batch.numpy().squeeze()
        labels = label_batch.numpy()

    plt.figure(figsize=(10,15))
    for n in range(30):
        plt.subplot(6,5,n+1)
        plt.imshow(images[n], cmap = plt.cm.binary)
        color = 'green' if np.argmax(ps[n]) == labels[n] else 'red'
        plt.title(class_names[str(np.argmax(ps[n])+1)], color=color)
        plt.axis('off')        
    plt.savefig('./prediction_result/predict_test_set.png')
        
        
# Save the trained model as a Keras model.
def save_model(model):
    saved_path = './{}.h5'.format('saved_model_1')
    model.save(saved_path)
    
    
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
def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image,0)
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
    axis_label = filtered(classes+1)
    fig,(ax1,ax2) = plt.subplots(figsize=(10,3),ncols=2)
    ax1.imshow(test_image,cmap=plt.cm.binary)
    ax2.set_title(str('Class Probability - ' + Path(filename).stem))
    ax2.barh(np.array(axis_label),probs[0])
    save_prediction_result = str('./prediction_result/'+ Path(filename).stem + '.png' )
    plt.savefig(save_prediction_result)


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

# Main function
if __name__ == '__main__':

    # Hyper parameters
    batch_size = 34
    image_size = (224, 224)
    EPOCHS = 15

    # Load the class names
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
    
    ans = query_yes_no("Build and Train the Classifier? ")
    
    if ans == 0:
        # Load the model
        saved_path = './{}.h5'.format('saved_model')
        model = tf.keras.models.load_model(saved_path, custom_objects={'KerasLayer': hub.KerasLayer})
    else:
        training_set, validation_set, test_set, dataset, dataset_info = load_dataset()
        # Create the training, validation and test batches
        total_training_samples = dataset_info.splits['validation'].num_examples
        training_batches = training_set.shuffle(total_training_samples//2).map(normalize_image).batch(batch_size).prefetch(1)
        validation_batches = validation_set.map(normalize_image).batch(batch_size).prefetch(1)
        test_batches = test_set.map(normalize_image).batch(batch_size).prefetch(1)
        # Build your network.
        num_classes = dataset_info.features['label'].num_classes
        model = duild_model(num_classes)
        # Train the model and validate
        model, train_history = train_model(model,EPOCHS)
        # Check some predictions to see visually how the model is performing
        plot_progress(train_history)
        # Print the loss and accuracy values achieved on the entire test set.
        loss, accuracy = evaluate(model)
        # Check some predictions to see visually how the model is performing
        plot_prediction(test_batches)
        # Save the trained model as a Keras model.
        save_model(model)    
    
    # Prediction 
    folder = './test_images/'
    filenames = load_images_path(folder)

    for image_path in filenames:
        prediction_result(image_path)

