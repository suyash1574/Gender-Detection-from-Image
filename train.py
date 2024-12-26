import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import os
import shutil

# Absolute paths to your datasets
man_path = r'C:\Users\zinju\OneDrive\Desktop\Final Projects\Gender-detection\Gender-Detection-master\main\dataset\man'
woman_path = r'C:\Users\zinju\OneDrive\Desktop\Final Projects\Gender-detection\Gender-Detection-master\main\dataset\woman'

# Create a new directory structure
dataset_path = r'C:\Users\zinju\OneDrive\Desktop\Final Projects\Gender-detection\Gender-Detection-master\main\dataset\combined'
os.makedirs(dataset_path + r'\man', exist_ok=True)
os.makedirs(dataset_path + r'\woman', exist_ok=True)

# Copy images to the new directory structure
for file_name in os.listdir(man_path):
    shutil.copy(os.path.join(man_path, file_name), os.path.join(dataset_path, 'man', file_name))

for file_name in os.listdir(woman_path):
    shutil.copy(os.path.join(woman_path, file_name), os.path.join(dataset_path, 'woman', file_name))

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Save the model
model.save('gender_detection_model.h5')
