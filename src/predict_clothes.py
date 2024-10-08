import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import argparse
import os
import matplotlib.pyplot as plt

# Define the class names for Fashion MNIST
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_path, target_size=(28, 28)):
    img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict(model, img_array):
    predictions = model.predict(img_array)
    return np.argmax(predictions), predictions

def visualize_prediction(img_path, prediction, class_name):
    # Load the original image
    original_img = image.load_img(img_path)
    plt.imshow(original_img)
    plt.title(f'Predicted: {class_name} (Class: {prediction})')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Fashion MNIST class.')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image to be predicted.')
    parser.add_argument('--model_path', type=str, default='../models/my_model.keras', help='Path to the saved model file.')
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        print(f"Image file {args.img_path} does not exist.")
    elif not os.path.exists(args.model_path):
        print(f"Model file {args.model_path} does not exist.")
    else:
        model = load_model(args.model_path)
        img_array = preprocess_image(args.img_path)
        prediction, predictions = predict(model, img_array)
        class_name = class_names[prediction]
        print(f'Predicted class: {prediction}, Class name: {class_name}')
        visualize_prediction(args.img_path, prediction, class_name)
