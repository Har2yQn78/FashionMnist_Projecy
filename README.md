# Fashion MNIST Classification

This project trains a neural network to classify images from the Fashion MNIST dataset. It also includes a script to make predictions on new images using the trained model.

## Project Structure

Fashion_MNIST_Project/ 
├── models/
│  └── my_model.keras 
├── src/ │ 
│  ├── train.py │ 
│  ├── predict_with_preprocessing.py │ 
│  ├── utils.py │ └── data/ 
├── requirements.txt 
├── README.md 
└── .gitignore

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Fashion_MNIST_Project.git
   cd Fashion_MNIST_Project
Create and activate a virtual environment:
   ```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install the dependencies:
```
pip install -r requirements.txt
```
To train the model, run the train.py script:
```
      python src/train.py
```
This will train the model, save it to the models directory

To make a prediction on a new image, use the predict_clothes_and_color.py script:
```
python src/predict_clothes_and_color.py --img_path path/to/your/image.jpg --clothes_model_path models/my_model.keras --color_model_path models/my_model_color.keras

```
To make a prediction on a new image (just predict the clothes), use the predict_clothes.py scripts:
```
python src/predict_clothes.py --img_path path/to/your/image.jpg --model_path models/my_model.keras
```
To make a prediction on a new image (just predict the color), use the predict_color.py scripts:
```
python src/predict_color.py --img_path path/to/your/image.jpg --model_path models/my_model_color.keras
```

Replace path_to_your_image with the path to the image you want to predict

This project is licensed under the MIT License - see the LICENSE file for details.
