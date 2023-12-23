# Handwritten digit classifier

Este proyecto es un clasificador de dígitos manuscritos utilizando una red neuronal simple en TensorFlow.

This project is a handwritten digit classifier using a simple neural network in TensorFlow.

## Dependencias - Dependencies

Este proyecto depende de las siguientes bibliotecas:
This project depends on the following libraries:

- TensorFlow
- TensorFlow Datasets
- Numpy
- Matplotlib
- Math
- Logging

## Conjunto de datos - Dataset

El conjunto de datos utilizado es el MNIST, que es un conjunto de dígitos manuscritos. El conjunto de datos se divide en un conjunto de entrenamiento y un conjunto de prueba.

The dataset used is the MNIST, which is a set of handwritten digits. The dataset is divided into a training set and a test set.

## Normalización - Normalization

Las imágenes en el conjunto de datos se normalizan para que los valores de los píxeles estén en el rango de 0 a 1.

The images in the dataset are normalized so that the pixel values are in the range 0 to 1.

## Modelo - Model

El modelo es una red neuronal simple con dos capas ocultas de 20 neuronas cada una y una capa de salida de 10 neuronas (una para cada dígito del 0 al 9). La función de activación utilizada en las capas ocultas es ReLU y en la capa de salida es Softmax.

The model is a simple neural network with two hidden layers of 20 neurons each and an output layer of 10 neurons (one for each digit from 0 to 9). The activation function used in the hidden layers is ReLU and in the output layer is Softmax.

## Entrenamiento - Training

El modelo se entrena con un tamaño de lote de 32 y durante 5 iteraciones.

The model is trained with a batch size of 32 and for 5 epochs.

## Evaluación - Evaluation

Después del entrenamiento, el modelo se evalúa en el conjunto de prueba.

After training, the model is evaluated on the test set.

## Visualización - Visualization

Finalmente, se visualizan las predicciones del modelo en un conjunto de imágenes de prueba. Para cada imagen, se muestra la imagen, la etiqueta predicha y la etiqueta verdadera. Si la predicción es correcta, la etiqueta se muestra en azul, de lo contrario, en rojo.
Se guardaron ejemplos de predicciones en la carpeta `predictions`.


Finally, the model's predictions are visualized on a set of test images. For each image, the image, the predicted label and the true label are shown. If the prediction is correct, the label is shown in blue, otherwise, in red.
Examples of predictions were saved in the `predictions` folder.

## Contribuciones - Contributions
Se aceptan contribuciones al proyecto. Para ello, se debe crear un pull request con los cambios propuestos. Si el pull request es aceptado, se incorporarán los cambios al proyecto.

Contributions to the project are accepted. To do this, you must create a pull request with the proposed changes. If the pull request is accepted, the changes will be incorporated into the project.