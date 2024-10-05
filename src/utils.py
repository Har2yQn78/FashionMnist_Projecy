import os
import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(img_path, target_size=(28, 28)):
    img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array
