import warnings
warnings.filterwarnings('ignore')

import json
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('path', action='store', help='path to image')
parser.add_argument('model', action='store', help='path to stored model')
parser.add_argument('--top_k', action='store', type=int, default=1, help='number of most probable classes')
parser.add_argument('--category_names', action='store', help='file which maps classes to names')
args = parser.parse_args()

image_size = 224


def process_image(image):
    im = Image.open(image)
    image = np.asarray(im)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()


try:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
except:
    pass


def predictLabel(image_path, model_pred, top_k=1):
    processed_test_image = process_image(image_path)

    ps = model_pred.predict(np.expand_dims(processed_test_image, axis=0))

    top_values, top_indices = tf.math.top_k(ps, top_k)
    print("These are the top propabilities", top_values.numpy()[0])
    top_classes = [class_names[str(value + 1)] for value in top_indices.cpu().numpy()[0]]
    print('Of these top classes', top_classes)
    return top_values.numpy()[0], top_classes


def predictClasses(image_path, model_pred, top_k=1):
    processed_test_image = process_image(image_path)

    ps = model_pred.predict(np.expand_dims(processed_test_image, axis=0))

    top_values, top_indices = tf.math.top_k(ps, top_k)
    print("These are the top propabilities", top_values.numpy()[0])
    top_classes = [value for value in top_indices.cpu().numpy()[0]]
    print('Of these top classes', top_classes)
    return top_values.numpy()[0], top_classes


model = tf.keras.models.load_model(str('./' + args.model), custom_objects={'KerasLayer': hub.KerasLayer})

img = (str(args.path))  # process image to pytensor using device

if (args.category_names != None):
    values, classes = predictLabel(img, model, args.top_k)
else:
    values, classes = predictClasses(img, model, args.top_k)