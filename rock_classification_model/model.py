import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers.legacy import Adam
from sklearn.metrics import classification_report

# Set the base path to the directory where the 'training_data', 'validation_data', and 'testing_data' are located
base_path = '.'  # Use '.' to refer to the current directory

# Define the paths to the training, validation, and testing datasets
train_dir = os.path.join(base_path, 'training_data')
val_dir = os.path.join(base_path, 'validation_data')
test_dir = os.path.join(base_path, 'testing_data')

# Image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 16

# Data generators for training and validation with data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

# Prepare the data generators
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_dir, target_size=(img_width, img_height),
                                                batch_size=batch_size, class_mode='categorical', shuffle=False)

# The number of classes (categories) can be determined from the training generator
num_categories = train_generator.num_classes

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_categories, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=10,
                    validation_data=val_generator, validation_steps=val_generator.samples // batch_size)

# Evaluate the model on the testing data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height),
                                                  batch_size=batch_size, class_mode='categorical', shuffle=False)

# Predict the test data
test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)
predictions = model.predict(test_generator, steps=test_steps_per_epoch)

# Compute the classification report
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Save the model
model.save(os.path.join(base_path, 'model.h5'))

# Output the accuracy
_, accuracy = model.evaluate(test_generator, steps=test_steps_per_epoch)
print(f'Test accuracy: {accuracy*100:.2f}%')

# After training the model, save the class indices to a file
import json

class_indices = train_generator.class_indices
class_indices_path = os.path.join(base_path, 'class_indices.json')

# Save the class indices
with open(class_indices_path, 'w') as class_indices_file:
    json.dump(class_indices, class_indices_file)
