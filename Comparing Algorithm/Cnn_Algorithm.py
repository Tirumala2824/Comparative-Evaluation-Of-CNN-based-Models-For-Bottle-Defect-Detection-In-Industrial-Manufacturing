import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import icecream as ic


# Define functions to train and record metrics

def train_and_record_metrics(model, train_generator, validation_generator, epochs, algorithm):
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Recording accuracy and loss values for training and validation
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create DataFrame to store accuracy and loss
    df = pd.DataFrame({'Epoch': range(1, epochs + 1),
                       'Train_Accuracy': train_accuracy,
                       'Val_Accuracy': val_accuracy,
                       'Train_Loss': train_loss,
                       'Val_Loss': val_loss})

    # Save accuracy and loss to CSV file
    df.to_csv(f'{algorithm}_metrics.csv', index=False)

    return history


# Path and hyperparameters
main_directory = 'D://python//Flask-2//Bottle_Fault_Detection//images'
img_width, img_height = 150, 150
batch_size = 32
epochs = 100
validation_split = 0.2

# Image data generator
data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=validation_split)

# Create generators for training and validation data
train_generator = data_generator.flow_from_directory(
    main_directory,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = data_generator.flow_from_directory(
    main_directory,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')
num_classes=9

print(train_generator.class_indices)
# Define CNN layers
cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and record metrics for CNN
cnn_history = train_and_record_metrics(cnn_model, train_generator, validation_generator, epochs, 'CNN')


# Test the CNN model
Cnn_test_loss, Cnn_test_accuracy = cnn_model.evaluate(validation_generator)
print(f"CNN Test Accuracy: {Cnn_test_accuracy}")
print(f"CNN Test Loss: {Cnn_test_loss}")


ann_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(img_width, img_height, 3)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and record metrics for ANN
ann_history = train_and_record_metrics(ann_model, train_generator, validation_generator, epochs, 'ANN')

# Test the ANN model
test_loss, test_accuracy = ann_model.evaluate(validation_generator)
print(f"ANN Test Accuracy: {test_accuracy}")
print(f"ANN Test Loss: {test_loss}")

# Visualizations
# Heatmap for CNN metrics
# Load metrics for CNN and ANN
cnn_metrics = pd.read_csv('CNN_metrics.csv')
ann_metrics = pd.read_csv('ANN_metrics.csv')

# Line plot for CNN and ANN accuracy comparison over epochs
plt.plot(cnn_metrics['Epoch'], cnn_metrics['Train_Accuracy'], label='CNN Train Accuracy')
plt.plot(cnn_metrics['Epoch'], cnn_metrics['Val_Accuracy'], label='CNN Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN Accuracy over Epochs')
plt.legend()
plt.show()

plt.plot(ann_metrics['Epoch'], ann_metrics['Train_Accuracy'], label='ANN Train Accuracy')
plt.plot(ann_metrics['Epoch'], ann_metrics['Val_Accuracy'], label='ANN Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('ANN Accuracy over Epochs')
plt.legend()
plt.show()

# Scatter plot for CNN vs ANN accuracy
plt.plot(cnn_metrics['Epoch'], cnn_metrics['Val_Accuracy'], label='CNN Accuracy')
plt.plot(ann_metrics['Epoch'], ann_metrics['Val_Accuracy'], label='ANN Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('CNN vs ANN Accuracy')
plt.legend()
plt.show()

# Bar plot for CNN vs ANN loss at last epoch
last_epoch = epochs
cnn_last_loss = cnn_metrics[cnn_metrics['Epoch'] == last_epoch]['Val_Loss'].values[0]
ann_last_loss = ann_metrics[ann_metrics['Epoch'] == last_epoch]['Val_Loss'].values[0]

plt.bar(['CNN', 'ANN'], [cnn_last_loss, ann_last_loss])
plt.xlabel('Algorithm')
plt.ylabel('Loss')
plt.title('CNN vs ANN Loss at Last Epoch')
plt.show()
