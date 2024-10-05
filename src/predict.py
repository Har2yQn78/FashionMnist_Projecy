import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import argparse
import pandas as pd

def load_model(model_path='../models/my_model.keras'):
    return tf.keras.models.load_model(model_path)

def predict(model, img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    return np.argmax(predictions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Fashion MNIST class.')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image to be predicted.')
    args = parser.parse_args()

    model = load_model()
    prediction = predict(model, args.img_path)
    print(f'Predicted class: {prediction}')
