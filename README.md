# Glomeruli-Image-Classification
## 1. Project Overview
We focus on developing a deep learning model to classify glomeruli images into two categories: globally sclerotic and non-globally sclerotic. The classification is performed using EfficientNetB3, a powerful convolutional neural network architecture that balances model size and performance.

## 2. Approach
EfficientNetB3 is chosen due to its balance between accuracy and efficiency. It leverages a compound scaling method that uniformly scales all dimensions of depth, width, and resolution using a set of fixed scaling coefficients. Given the problem statement, EfficientNetB3 allows us to achieve better performance with fewer parameters compared to other architectures.

## 3. Pre-processing Steps
### 3.1. Image Loading and Conversion:
Images are loaded from specified directories and converted to RGB format.
### 3.2. Resizing and Padding:
Images are resized and padded to a uniform size of 300x300 pixels (to satisfy the constraints of EfficientNetB3) while retaining maximum information.
### 3.3. Data Augmentation:
Additional data is generated through transformations including random scaling, translation, and noise addition to prevent overfitting.
### 3.4. Normalization: 
Image pixel values are normalized to the range [0, 1] by dividing by 255.

## 4. Post-processing Steps
### 4.1. Stratified Split:
The dataset is split into training and testing sets using stratified sampling to maintain the balance of classes.
### 4.2. Model Freezing and Fine-tuning:
Initial training is done with the base EfficientNetB3 layers frozen.
Fine-tuning involves unfreezing the top layers and training with a lower learning rate to improve model performance.

## 5. Performance Metrics
The model is evaluated using:
### 5.1. Accuracy: 
The proportion of correctly classified images.
### 5.2. Loss: 
Binary cross-entropy loss to measure the error in classification.
### 5.3. Precision: 
The ratio of correctly predicted positive observations to the total predicted positives.
### 5.4. Recall: 
The ratio of correctly predicted positive observations to all observations in the actual class.
### 5.5. Confusion Matrix:
A confusion matrix provides a detailed breakdown of correct and incorrect predictions for each class.

## 6. Running the Code
### 6.1. Ensure you have the following libraries installed:
1. TensorFlow
2. OpenCV
3. NumPy
4. h5py
5. scikit-learn
6. matplotlib
7. zipfile
8. gc
### 6.2. Execution Steps:
#### 6.2.1. Method-1: Building Dataset From Scratch, and then using it to train and fine tune the Deep Learning Model, through Google Colab/Kaggle
1. Upload the images dataset (Zip file Recommended) in Google Drive to access it in Google Colab/Kaggle if google colab services are not enough.
2. Upload the jupyter notebook file([Datagen_b3modeltrain.ipynb](Datagen_b3modeltrain.ipynb)) to your Google Colab/Kaggle.
3. Follow the cells in the notebook to execute the preprocessing steps, model training, and fine-tuning.
a. Import the required libraries.
b. Execute the functions to convert images to vectors, resize, and pad them.
c. Perform data augmentation and save the processed data into an HDF5 file.
d. Load the data from the HDF5 file and normalize it.
e. Split the data into training and testing sets.
f. Define and compile the EfficientNetB3 model.
g. Train the model with frozen layers and save it.
h. Unfreeze the top layers, fine-tune the model, and save the final model.
#### 6.3.2. Method-2: Using the pre built dataset (built based on images dataset) directly and Training the deep learning Model
1. Upload the Dataset (<a href = "https://www.kaggle.com/datasets/gaddamakhilreddy/preprocessed-glomeruli-dataset">.h5 file</a>) in Google Drive to access it in google colab/ in kaggle and access it in Kaggle.
2. Upload the jupyter notebook file[Model.ipynb](Model.ipynb) to your Google Colab/Kaggle.
3. Follow the cells in the notebook to Train the Deep Learning Model.
#### 6.3.3. Executing evaluation.py:
1. Do not forget to download the Trained <a href="https://drive.google.com/file/d/11uOJ0DyONX64G1vd6b6q56f3MEC1FQXR/view?usp=sharing">EfficientNetB3</a> model and make sure to include it in your working directory while executing evaluation.py. To execute evaluation.py, you can use any IDE/Open source environments like vscode(recommended)/Google Colab/Kaggle.
