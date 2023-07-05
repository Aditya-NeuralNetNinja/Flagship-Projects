# Siamese Neural Network for MNIST Digit Verification 👥🔢

The goal of this project is to build a Siamese Neural Network that can verify whether two digits, represented as images from the MNIST dataset, are of the same class or not. The model is trained using the contrastive loss function and the efficacy of the model is evaluated using a subset of the MNIST test dataset.

## Table of Contents 📚

- [Project Setup](#project-setup)
- [Model Building](#model-building)
- [Data Preparation](#data-preparation)
- [Model Compilation](#model-compilation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Image Comparison](#image-comparison)

## Project Setup 🚀

First, load the MNIST digits dataset.

```python
import tensorflow as tf  # Import TensorFlow library

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Output the shapes of training and testing data
X_train.shape, y_train.shape, X_test.shape, y_test.shape
```

## Model Building 🛠️

Next, build a Siamese Neural Network using TensorFlow and Keras.

```python
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Reshape, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential

# ... Refer to the provided code for full model details ...

# Create the final model
model = Model(inputs=[img_A_inp, img_B_inp], outputs=output)

# Print a summary of the model
model.summary()
```

## Data Preparation 🎲

This includes random sampling and dimension confirmation of training images and labels, generating pairs of images and labels to train the Siamese network, and random sampling of test images and labels for evaluation.

```python
import numpy as np

# ... Refer to the provided code for full details on data preparation ...

# Print the shapes of the sampled test images and labels to confirm their dimensions
X_test_sample.shape, y_test_sample.shape
```

## Model Compilation 🔄

Compile the model using binary crossentropy as the loss function and Adam optimizer.

```python
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])
```

## Model Training 💪

Train the Siamese model using the prepared dataset.

```python
from tensorflow.keras.callbacks import EarlyStopping

# If the model's validation loss doesn't improve for 3 consecutive epochs, 
# the training will be stopped to prevent overfitting
es = EarlyStopping(patience=3)

model.fit(
    # Feed the pairs of training images into the model
    # X_train_pairs[:, 0, :, :] represents the first image in each pair
    # X_train_pairs[:, 1, :, :] represents the second image in each pair
    [X_train_pairs[:, 0, :, :], X_train_pairs[:, 1, :, :]],
          y=y_train_pairs,
          validation_split=0.3, 
          epochs=100,
          batch_size=32,
          callbacks=[es])
```

## Model Evaluation 👀

Evaluate the model's efficacy by testing it on unseen data.

```python
# Select the first and eighteenth image from the test dataset
img_A, img_B = X_test[0], X_test[17]

# Get the corresponding labels of the selected images
label_A, label_B = y_test[0], y_test[17]

#

Print the labels of the selected images
```python
label_A, label_B
```

Next, display the selected images.
```python
import matplotlib.pyplot as plt

# Create new figure with set DPI=28, as MNIST image =(28,28)
plt.figure(dpi=28)

# Display img_A
plt.imshow(img_A)

plt.figure(dpi=28)

# Display img_B
plt.imshow(img_B)
```

## Image Comparison 📸

Finally, let's compare the images using our trained Siamese Neural Network. We aim to predict if the two selected images are of the same class or not.
Images belong to same class '7' below:

![image](https://github.com/Aditya-NeuralNetNinja/Flagship-Projects/assets/108260519/d32d7b8c-05e5-4d27-903c-afe5d6966b78)


```python
# Predict if img_A and img_B are from the same class (True if prediction > 0.5)
model.predict([img_A.reshape((1, 28, 28)), 
               img_B.reshape((1, 28, 28))]).flatten()[0] > 0.5
```

As the predicted value is greater than 0.5, the output is True, indicating that the model believes the images are of the same class. 🎉

This wraps up the usage guide for this project. The Siamese Network is a powerful tool for image comparison tasks, as demonstrated with the MNIST dataset. Happy coding! 🚀

