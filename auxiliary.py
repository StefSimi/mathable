import os
from PIL import Image
import cv2 as cv
import numpy as np

# Define the input and output folders
input_folder = "mathable_templates"  # Replace with the path to your folder with 100x100 images
output_folder = "modified_templates"  # Replace with the desired output folder path
output_folder2 = "modified_templates_binary"  # Replace with the desired output folder path

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)

    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            # Open the image
            with Image.open(input_path) as img:
                # Resize the image to 32x32 pixels
                resized_img = img.resize((95, 95))

                # Convert the PIL image to a NumPy array
                img_array = np.array(resized_img)

                # Convert to grayscale if necessary (threshold requires a single channel)
                if img_array.ndim == 3:
                    img_array = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)

                # Apply the binary threshold
                _, threshold = cv.threshold(img_array, 127, 255, cv.THRESH_BINARY)

                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)

                # Save the binary threshold image to the second output folder
                output_path2 = os.path.join(output_folder2, filename)
                cv.imwrite(output_path2, threshold)

                print(f"Resized and saved: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        print(f"Skipping non-image file: {filename}")
