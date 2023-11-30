# Import necessary libraries
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np

# Function to load and preprocess CIFAR-10 dataset
def load_and_preprocess_data():
    # Load CIFAR-10 dataset
    (training_images, training_labels), (X_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    training_images = training_images.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    # One-hot encode the target variable
    training_labels = to_categorical(training_labels, 10)
    y_test = to_categorical(y_test, 10)

    # Split the data into training and validation sets
    training_images, validation_images, training_labels, validation_labels = train_test_split(
        training_images, training_labels, test_size=0.2, random_state=42
    )

    return training_images, training_labels, validation_images, validation_labels, X_test, y_test

# Function to display sample images
def display_sample_images(images, labels, class_names, num_samples=5, title_prefix=""):
    plt.figure(figsize=(10, 3))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(images[i])
        plt.title(f"{title_prefix}{class_names[np.argmax(labels[i])]}")
        plt.axis('off')
    plt.show()

# Function to build the improved CNN model
def build_improved_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model

# Function to train the improved model
def train_improved_model(model, training_images, training_labels, validation_data, epochs=50, batch_size=64):
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Learning rate scheduler
    def lr_schedule(epoch):
        return 1e-3 * 0.9 ** epoch
    
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    datagen.fit(training_images)

    history = model.fit(
        datagen.flow(training_images, training_labels, batch_size=batch_size),
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[lr_scheduler]
    )

    return history

# Function to evaluate and display sample images
def evaluate_and_display(model, X_test, y_test, class_names):
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Loss: {loss}")
    print(f"Final Test Accuracy: {accuracy}")

    # Display actual and predicted sample testing images
    predictions = model.predict(X_test[:5])
    display_sample_images(X_test[:5], y_test[:5], class_names, title_prefix="Actual: ")
    display_sample_images(X_test[:5], predictions, class_names, title_prefix="Predicted: ")


# Main part of the script
if _name_ == "_main_":
    # Load and preprocess the data
    training_images, training_labels, validation_images, validation_labels, X_test, y_test = load_and_preprocess_data()

    # Display sample training images
    class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    display_sample_images(training_images, training_labels, class_names)

    # Build the improved CNN model
    improved_cnn_model = build_improved_cnn_model()

    # Train the improved model
    train_improved_model(improved_cnn_model, training_images, training_labels, 
                         validation_data=(validation_images, validation_labels))

    # Evaluate and display results
    evaluate_and_display(improved_cnn_model, X_test, y_test, class_names)
