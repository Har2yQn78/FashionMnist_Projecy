import tensorflow as tf
import numpy as np
import os
import argparse
from tensorflow.keras.models import load_model as keras_load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import predict_with_preprocessing

# Define the color names
label_names = ["black", "blue", "brown", "green", "grey", "orange", "red", "violet", "white", "yellow"]

# Load the color prediction model
def load_color_model(model_path):
    return keras_load_model(model_path)

# Preprocess image for color prediction
def preprocess_image_color(img_path, target_size=(64, 64)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict color
def predict_color(model, img_array):
    predictions = model.predict(img_array)
    return np.argmax(predictions), predictions

# Visualize prediction
def visualize_prediction(img_path, cloth_prediction, cloth_class_name, color_prediction, color_name):
    original_img = image.load_img(img_path)
    plt.imshow(original_img)
    plt.title(f'Predicted: {cloth_class_name} (Class: {cloth_prediction}), Color: {color_name} (Class: {color_prediction})')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Fashion MNIST class and color.')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image to be predicted.')
    parser.add_argument('--clothes_model_path', type=str, required=True, help='Path to the saved clothes model file.')
    parser.add_argument('--color_model_path', type=str, required=True, help='Path to the saved color model file.')
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        print(f"Image file {args.img_path} does not exist.")
    elif not os.path.exists(args.clothes_model_path):
        print(f"Clothes model file {args.clothes_model_path} does not exist.")
    elif not os.path.exists(args.color_model_path):
        print(f"Color model file {args.color_model_path} does not exist.")
    else:
        # Predict clothes type
        clothes_model = predict_with_preprocessing.load_model(args.clothes_model_path)
        clothes_img_array = predict_with_preprocessing.preprocess_image(args.img_path)
        cloth_prediction, _ = predict_with_preprocessing.predict(clothes_model, clothes_img_array)
        cloth_class_name = predict_with_preprocessing.class_names[cloth_prediction]

        # Predict color
        color_model = load_color_model(args.color_model_path)
        color_img_array = preprocess_image_color(args.img_path)
        color_prediction, _ = predict_color(color_model, color_img_array)
        color_name = label_names[color_prediction]

        print(f'Predicted class: {cloth_class_name}, Color: {color_name}')
        visualize_prediction(args.img_path, cloth_prediction, cloth_class_name, color_prediction, color_name)
