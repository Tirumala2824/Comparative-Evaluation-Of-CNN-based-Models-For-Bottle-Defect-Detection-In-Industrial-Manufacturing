import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Replace 'path_to_dataset_folder' with the path to your dataset folder containing subfolders for each class
dataset_folder = 'D://python//Flask-2//Bottle_Fault_Detection//images'
classes = os.listdir(dataset_folder)

# Set up a figure to display images
plt.figure(figsize=(12, 8))

for i, class_name in enumerate(classes, 1):
    class_folder = os.path.join(dataset_folder, class_name)
    class_images = os.listdir(class_folder)
    
    # Choose one image from each class
    image_path = os.path.join(class_folder, class_images[0])  # Choose the first image from the class
    
    # Load and display the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format for Matplotlib
    
    plt.subplot(3, 3, i)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis('off')

plt.tight_layout()
plt.show()




# # Replace 'path_to_dataset_folder' with the path to your dataset folder containing subfolders for each class
# dataset_folder = 'D://python//Flask-2//Bottle_Fault_Detection//images'
# classes = os.listdir(dataset_folder)

# Image preprocessing parameters
target_size = (128, 128)

# Image augmentation
data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Initialize lists to store images and corresponding class names
resized_images = []
normalized_images = []
augmented_images = []
class_names = []

# Loop through each class folder and process images
for class_name in classes:
    class_folder = os.path.join(dataset_folder, class_name)
    class_images = [img for img in os.listdir(class_folder) if img.endswith('.jpg') or img.endswith('.png')]
    
    if len(class_images) > 0:
        # Choose one image from each class
        image_path = os.path.join(class_folder, class_images[0])  # Choose the first image from the class
        class_names.append(class_name)  # Store class name
        
        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
        
        # Resize the image
        resized_image = cv2.resize(image, target_size)
        resized_images.append(resized_image)
        
        # Normalize the image
        normalized_image = resized_image / 255.0  # Normalizing pixel values
        normalized_images.append(normalized_image)
        
        # Perform augmentation on the image
        img = np.expand_dims(resized_image, axis=0)  # Expand dimensions to fit into ImageDataGenerator
        augmented = data_generator.flow(img, batch_size=1)
        augmented_image = augmented.next()[0].astype(np.float32)
        augmented_images.append(augmented_image)

# Visualize resized images for each class
plt.figure(figsize=(12, 6))
for i, (image, class_name) in enumerate(zip(resized_images, class_names), 1):
    plt.subplot(3, 3, i)
    plt.imshow(image)
    plt.title(f'Resized - {class_name}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Visualize normalized images for each class
plt.figure(figsize=(12, 6))
for i, (image, class_name) in enumerate(zip(normalized_images, class_names), 1):
    plt.subplot(3, 3, i)
    plt.imshow(image)
    plt.title(f'Normalized - {class_name}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Visualize augmented images for each class
plt.figure(figsize=(12, 6))
for i, (image, class_name) in enumerate(zip(augmented_images, class_names), 1):
    plt.subplot(3, 3, i)
    plt.imshow(image)
    plt.title(f'Augmented - {class_name}')
    plt.axis('off')
plt.tight_layout()
plt.show()

