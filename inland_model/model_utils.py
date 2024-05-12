import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def build_and_compile_model(x_train):
    """
    Function to build and compile a neural network model.

    Parameters:
    x_train (numpy.ndarray): Training data features.

    Returns:
    keras.Model: Compiled neural network model.
    """
    # Normalize the input features
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(x_train))

    # Define the neural network architecture
    model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    # Compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    
    return model

def plot_loss(history):
    """
    Function to plot the training and validation loss over epochs.

    Parameters:
    history (keras.callbacks.History): Training history of the model.
    """
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.autoscale(enable=True, axis='y')
    plt.xlabel('Epoch')
    plt.ylabel('Error [Flood Depth]')
    plt.legend()
    plt.grid(True)

def predict(model, x_test, y_test):
    """
    Function to plot the true flood depth against the predicted flood depth.

    Parameters:
    model (keras.Model): Trained neural network model.
    x_test (numpy.ndarray): Testing data features.
    y_test (numpy.ndarray): True flood depths for the testing data.
    """
    # Generate predictions
    test_predictions = model.predict(x_test).flatten()

    # Plot true vs. predicted flood depths
    plt.scatter(y_test, test_predictions)
    plt.xlabel('True [Flood Depth]')
    plt.ylabel('Predictions [Flood Depth]')
    
    # Calculate the minimum and maximum values for both axes
    min_val = min(np.min(y_test), np.min(test_predictions))
    max_val = max(np.max(y_test), np.max(test_predictions))
    
    # Set axis limits
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Plot the identity line (y = x)
    plt.plot([min_val, max_val], [min_val, max_val], 'r')
