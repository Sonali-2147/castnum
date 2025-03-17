import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Define the Enhanced CNN Model
model = Sequential([
    Input(shape=(64, 64, 1)),  # Explicit input layer
    
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Flatten and Fully Connected Layers
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),  # Prevent overfitting
    Dense(128, activation='relu'),
    Dropout(0.3),
    
    Dense(3, activation='softmax')  # Assuming 3 shape categories (Circle, Triangle, Rectangle)
])

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the Model
model.save("shape_classifier.h5")

print("Improved Model saved successfully!")
