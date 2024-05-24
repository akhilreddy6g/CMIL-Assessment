import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from Dataset_Generation import image_vector, reshape_image_size


# Function to convert every image present in a folder (with path "files_path") to a vector
def image_vector(files_path):
    # Storing all the names of image files (present at "files_path" location) in "files"
    files = os.listdir(files_path)
    # Appending the "files_path" name along with the names of image files present in "files"
    if files_path[-1]=="/":
        files_complete_path = [files_path + i for i in files]
    else:    
        files_complete_path = [files_path + "/" + i for i in files]
    image_vector_list = []
    for i in files_complete_path:
        image = cv2.imread(i)
        # Convert BGR (default in OpenCV) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_vector_list.append(image_rgb)
        # Appending Shape of the Vector of all the Images
    return image_vector_list, files

def reshape_image_size(vector_list, target_height, target_width):
    # target_size is a list of 2 numbers
    list_copy = []
    for vector in vector_list:
        height, width = vector.shape[0], vector.shape[1]
        # Resize the image based on target_size:
        if height>=target_height and width>=target_width:
            resized_vector = cv2.resize(vector, (target_width, target_height), interpolation=cv2.INTER_AREA)
        elif height>=target_height and width<target_width:
            resized_vector = cv2.resize(vector, (width, target_height), interpolation=cv2.INTER_AREA)
        elif height<target_height and width>=target_width:
            resized_vector = cv2.resize(vector, (target_width, height), interpolation=cv2.INTER_AREA)
        else:
            resized_vector = cv2.resize(vector, (width, height), interpolation=cv2.INTER_AREA)
        # Calculate padding to reach target size
        pad_width = target_width - resized_vector.shape[1]
        pad_height = target_height - resized_vector.shape[0]
        top, bottom = pad_height // 2, pad_height - (pad_height // 2)
        left, right = pad_width // 2, pad_width - (pad_width // 2)
        # Setting padding, i.e., adding zeros along height or width, which are less target_size
        padded_vector = cv2.copyMakeBorder(resized_vector, top, bottom, left, right, cv2.BORDER_CONSTANT)
        list_copy.append(padded_vector)
    list_copy = np.array(list_copy)
    return list_copy

def evaluate_images(folder_path):
    # Load the saved model
    model = load_model('final-b3ic.keras')
    # Get image vectors and filenames
    image_vectors, imagefilenames = image_vector(folder_path)
    # Reshape images to match model input size
    resized_image_vectors = reshape_image_size(image_vectors, 300, 300)
    # Normalize image vectors
    resized_image_vectors = resized_image_vectors / 255.0
    # Predict classes for image vectors
    predictions = model.predict(resized_image_vectors)
    # Round predictions to get class labels
    predicted_classes = (predictions > 0.5).astype(int)
    # Create a DataFrame to store filename and predicted class
    evaluation_df = pd.DataFrame({'Filename': imagefilenames, 'Predicted Class': predicted_classes.flatten()})
    # Save DataFrame to CSV
    evaluation_df.to_csv('evaluation.csv', index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate glomeruli image patches using a trained deep learning model.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing glomeruli image patches.')
    args = parser.parse_args()
    evaluate_images(args.folder_path)
