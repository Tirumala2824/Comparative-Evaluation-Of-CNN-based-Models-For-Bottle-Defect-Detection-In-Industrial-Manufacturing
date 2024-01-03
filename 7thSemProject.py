# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# import os
#
#
# # Function to load images from multiple subfolders and assign labels
# def load_images_from_folders(root_directory):
#     all_images = []
#     all_labels = []
#
#     # Traverse through subfolders and process images
#     for root, dirs, files in os.walk(root_directory):
#         for filename in files:
#             if filename.endswith(('.jpg', '.jpeg', '.png')):  # Assuming images have these extensions
#                 image_path = os.path.join(root, filename)
#                 label = os.path.basename(root)  # Use the subfolder name as the label
#
#                 # Read and preprocess the image
#                 img = cv2.imread(image_path)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format if needed
#                 img = cv2.resize(img, (100, 100))  # Resize the image to 100x100
#
#                 all_images.append(img)
#                 all_labels.append(label)
#
#     return np.array(all_images), np.array(all_labels)
#
#
#
# # Example root directory containing subfolders with images
# root_directory = 'D://python//Flask-2//Bottle_Fault_Detection//images'
#
# # Image resizing
# def resize_images(images, width, height):
#     resized_images = []
#     for img in images:
#         resized = cv2.resize(img, (width, height))
#         resized_images.append(resized)
#     return np.array(resized_images)
#
#
# # Normalization
# def normalize_images(images):
#     normalized_images = images.astype('float32') / 255.0  # Scale pixel values between 0 and 1
#     return normalized_images
#
#
# # Augmentation (using OpenCV for simple example)
# def augment_images(images):
#     augmented_images = []
#     for img in images:
#         # Example augmentation: flipping horizontally
#         flipped_img = cv2.flip(img, 1)
#         augmented_images.append(flipped_img)
#     return np.array(augmented_images)
#
#
# # Data Split
# def split_data(images, labels, test_size=0.2, val_size=0.2, random_state=42):
#     # Split into train and test sets
#     x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state)
#
#     # Further split train set into train and validation sets
#     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=random_state)
#
#     return x_train, x_val, x_test, y_train, y_val, y_test
#
#
# # Load images and labels from subfolders
# images, labels = load_images_from_folders(root_directory)
#
# # Resize images to 100x100
# resized_images = resize_images(images, 100, 100)
#
# # Normalize images
# normalized_images = normalize_images(resized_images)
#
# # Augment images
# augmented_images = augment_images(normalized_images)
#
# # Split data into train, validation, and test sets
# x_train, x_val, x_test, y_train, y_val, y_test = split_data(augmented_images, labels)
# from sklearn.preprocessing import LabelEncoder
#
# # Filter out labels present in the test set but not in the training or validation sets
# valid_labels = np.unique(np.concatenate((y_train, y_val)))
# y_test_filtered = np.array([label if label in valid_labels else 'unknown_label' for label in y_test])
#
# # Initialize a LabelEncoder and fit on all available labels (train, validation, and filtered test)
# label_encoder = LabelEncoder()
# label_encoder.fit(np.concatenate((y_train, y_val, y_test_filtered)))
#
# # Apply label encoding consistently across all splits
# y_train_encoded = label_encoder.transform(y_train)
# y_val_encoded = label_encoder.transform(y_val)
# y_test_encoded = label_encoder.transform(y_test_filtered)
#
# import tensorflow as tf
# from tensorflow.keras import layers, models
#
# # Define the CNN model architecture
# def create_cnn_model(input_shape, num_classes):
#     model = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dense(num_classes, activation='softmax')  # Use 'softmax' for multi-class classification
#     ])
#     return model
#
# # Set the input shape and the number of classes
# input_shape = (100, 100, 3)  # Adjust dimensions to match your image size
# num_classes = len(np.unique(labels))  # Assuming unique labels in your dataset
#
# # Create the CNN model
# model = create_cnn_model(input_shape, num_classes)
#
# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(x_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(x_val, y_val))
#
# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded)
# print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')



import os
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to the dataset
root_directory = 'D://python//Flask-2//Bottle_Fault_Detection//images'

# Load images and labels from subfolders
def load_images_from_folders(directory):
    images = []
    labels = []
    for folder in os.listdir(directory):
        for file in os.listdir(os.path.join(directory, folder)):
            image = tf.keras.preprocessing.image.load_img(os.path.join(directory, folder, file), target_size=(100, 100))
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(image)
            labels.append(folder)
    return np.array(images), np.array(labels)

images, labels = load_images_from_folders(root_directory)

# Initialize a LabelEncoder and fit on all available labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Apply label encoding consistently across all labels
encoded_labels = label_encoder.transform(labels)

# Define the CNN model architecture
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax') # Use 'softmax' for multi-class classification
    ])
    return model

# Set the input shape and the number of classes
input_shape = (100, 100, 3) # Adjust dimensions to match your image size
num_classes = len(np.unique(labels)) # Assuming unique labels in your dataset

# Create the CNN model
model = create_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Use 'sparse_categorical_crossentropy' for integer labels
              metrics=['accuracy'])

# Split data into train, validation, and test sets
x_train, x_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Further split train set into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Train the model for 10 epochs
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')