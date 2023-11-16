import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy


# Define your dataset directories
train_dir = 'train_data'
test_dir = 'test_data'

# Image dimensions
image_height = 100
image_width = 100
batch_size = 32

# Create data generators for training and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')  # Assuming 4 currency classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
num_epochs = 10
history = model.fit(
    train_generator,
    epochs=num_epochs,
    verbose=2
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
print(f'Test accuracy: {test_accuracy}')

# Save the model for future use
model.save('currency_recognition_model.h5')

# Inference
# Load the model and use it to make predictions on new currency images
loaded_model = keras.models.load_model('currency_recognition_model.h5')
predictions = loaded_model.predict(test_generator)

# Print predictions
print(predictions)


# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np

# # Load your trained model
# model = keras.models.load_model('currency_recognition_model.h5')

# # Define a function to predict currency from an image
# def predict_currency(image_path):
#     # Load and preprocess the image
#     img = load_img(image_path, target_size=(100, 100))  # Load the image and resize it to match the model's input size
#     img = img_to_array(img) / 255.0  # Convert image to array and normalize

#     # Make a prediction
#     prediction = model.predict(np.array([img]))
    
#     # Get the class label with the highest probability
#     class_labels = ['fifty','thousand', 'twenty', 'ten', 'fivehundred', 'hundred', ]
#     predicted_class = class_labels[np.argmax(prediction)]
#     model.summary()
#     # print(train_generator.class_indices)


#     return predicted_class

# # Example usage:
# image_path = r"C:\Users\SYED JAVITH\Downloads\currency-recognition-master\currency-recognition-master\39.jpg"
# predicted_currency = predict_currency(image_path)
# print(f'The detected currency is: {predicted_currency}')
