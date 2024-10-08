import os
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict clothes and color.')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image to be predicted.')
    parser.add_argument('--clothes_model_path', type=str, required=True, help='Path to the saved clothes model file.')
    parser.add_argument('--color_model_path', type=str, required=True, help='Path to the saved color model file.')
    args = parser.parse_args()

    print("Starting combined prediction...")

    if not os.path.exists(args.img_path):
        print(f"Image file {args.img_path} does not exist.")
    elif not os.path.exists(args.clothes_model_path):
        print(f"Clothes model file {args.clothes_model_path} does not exist.")
    elif not os.path.exists(args.color_model_path):
        print(f"Color model file {args.color_model_path} does not exist.")
    else:
        try:
            # Predict clothes
            print("Predicting clothes...")
            clothes_result = subprocess.run(
                ['python', 'src/predict_clothes.py', '--img_path', args.img_path, '--model_path',
                 args.clothes_model_path],
                capture_output=True, text=True, check=True
            )
            print("Clothes prediction output:")
            print(clothes_result.stdout)
            print(clothes_result.stderr)

            # Predict color
            print("Predicting color...")
            color_result = subprocess.run(
                ['python', 'src/predict_color.py', '--img_path', args.img_path, '--model_path', args.color_model_path],
                capture_output=True, text=True, check=True
            )
            print("Color prediction output:")
            print(color_result.stdout)
            print(color_result.stderr)

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running subprocess: {e}")
            print(f"Output: {e.output}")
            print(f"Error: {e.stderr}")
