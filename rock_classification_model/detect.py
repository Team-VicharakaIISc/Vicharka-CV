import json
import sys
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

# Function to load and prepare the image in the right format
def load_and_prepare_image(image_path, img_width=150, img_height=150):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image as we did before
    return img_array

# Load class indices
class_indices_path = './class_indices.json'  # Replace with your actual path
with open(class_indices_path, 'r') as class_indices_file:
    class_indices = json.load(class_indices_file)

# Reverse the class indices to get a mapping from index to class name
index_to_class = {v: k for k, v in class_indices.items()}

# Load the model
model_path = './model.h5'  # Replace with your actual path
model = load_model(model_path)

# Get image path from command line argument
image_path = sys.argv[1]

# Predict the rock type
img_array = load_and_prepare_image(image_path)
prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction, axis=1)[0]
predicted_class_name = index_to_class[predicted_class_index]

# Output the predicted rock type
print(f"This rock is predicted to be a: {predicted_class_name}")
