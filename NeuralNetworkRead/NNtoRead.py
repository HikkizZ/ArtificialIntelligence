# Libraries
#¡ Tensorflow Library
import tensorflow as tf
import tensorflow_datasets as tfds

#¡ Library for mathematical calculations and data visualization
import numpy as np
import matplotlib.pyplot as plt
import math

#¡ Library for event logging
import logging

# Get the logger
logger = tf.get_logger()
# Set the logger to only display errors
logger.setLevel(logging.ERROR)

#¡ Import the dataset
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)

#? 85% of the data will be used for training and 15% for testing
ds_train, ds_test = dataset['train'], dataset['test']

#? The number of images in the training and test sets is obtained
class_names  = [
    'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro',
    'Cinco', 'Seis', 'Siete', 'Ocho', 'Nueve'
]

# Obtain the number of images in the training and test sets
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

#! Normalize the data (0-255) to (0-1). (Pixel range)
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255 #? 255 is the maximum value of a pixel
    return images, labels

ds_train = ds_train.map(normalize) #? Normalize the training set
ds_test = ds_test.map(normalize) #? Normalize the test set

#¡ Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), #? Input layer
    tf.keras.layers.Dense(20, activation=tf.nn.relu), #? Hidden layer
    tf.keras.layers.Dense(20, activation=tf.nn.relu), #? Hidden layer
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) #? Output layer
])

model.compile(
    optimizer='adam', #? Algorithm to update the weights
    loss='sparse_categorical_crossentropy', #? Loss function
    metrics=['accuracy'] #? Metric to evaluate the model
)

#¡ Batch learning of 32 each
BATCH_SIZE = 32
ds_train = ds_train.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
ds_test = ds_test.batch(BATCH_SIZE)

#¡ Train the model
model.fit(
    ds_train,
    epochs=5, #? Number of iterations
    steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE) #? Number of steps per epoch
)

#¡ Evaluate the model
test_loss, test_accuracy = model.evaluate(
    ds_test,
    steps=math.ceil(num_test_examples/32) #? Number of steps
)

#¡ Results
print(f'Results of the model: {test_accuracy}')

#¡ Graph of the results

#? Take a batch of test images
for test_images, test_labels in ds_test.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
    
#¡ Function to plot the image
def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img[...,0], cmap=plt.cm.binary) #? Show the image
    
    predicted_label = np.argmax(predictions_array) #? Get the predicted label
    
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel(
        f'{class_names[predicted_label]} {100*np.max(predictions_array):2.0f}% ({class_names[true_label]})',
        color=color
    )

#¡Function to plot the graph
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    

#? Number of rows and columns to plot
num_rows = 5
num_cols = 6
#? Total number of images to plot
num_images = num_rows*num_cols
#? Create a figure
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

#¡ For each image
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1) #? Create a subplot
    plot_image(i, predictions, test_labels, test_images) #? Plot the image
    plt.subplot(num_rows, 2*num_cols, 2*i+2) #? Create a subplot
    plot_value_array(i, predictions, test_labels) #? Plot the graph
    
#? Show the figure
plt.show()