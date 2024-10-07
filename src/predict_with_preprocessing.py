import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import argparse

# Load the models
color_model = load_model(r'C:\Users\HARRY\Desktop\FashionMNIST_Project\models\my_model_color.keras')
clothing_model = load_model(r'C:\Users\HARRY\Desktop\FashionMNIST_Project\models\my_model.keras')

# Define class names
color_classes = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'red', 'violet', 'white', 'yellow']
clothing_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                    'Ankle boot']


def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def predict(img_path):
    # Preprocess image for color detection model
    color_img_array = preprocess_image(img_path, target_size=(64, 64))

    # Predict color
    color_pred = color_model.predict(color_img_array)
    color_class = np.argmax(color_pred)
    color_name = color_classes[color_class]

    # Preprocess image for clothing model
    clothing_img_array = preprocess_image(img_path, target_size=(28, 28))

    # Predict clothing item
    clothing_pred = clothing_model.predict(clothing_img_array)
    clothing_class = np.argmax(clothing_pred)
    clothing_name = clothing_classes[clothing_class]

    return color_name, clothing_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict color and clothing item.')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image file')
    args = parser.parse_args()

    color, clothing = predict(args.img_path)
    print(f"Predicted color: {color}")
    print(f"Predicted clothing item: {clothing}")

    # Visualization
    img = image.load_img(args.img_path)
    plt.imshow(img)
    plt.title(f"Color: {color}, Item: {clothing}")
    plt.show()
