from lib.helper import csv_parser, plot_series
from lib.libs import split_train_valid, windowed_dataset
from settings import window_size, batch_size, shuffle_buffer_size, split_size, lr

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def create_dataset():
    """parse dataset from file location and split dataset in training and validation set"""
    series, time = csv_parser()
    x_train, x_valid, time_train, time_valid = split_train_valid(split_size, series, time)
    return x_train, x_valid, time_train, time_valid


def create_model():
    """3 layers neural network model using relu activation function"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    return model


def learning_rate_scheduler(model, dataset):
    """learning rate scheduler implemented on top of dense model and return trained model"""
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10 ** (epoch / 20)
    )
    optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss="mse", optimizer=optimizer)
    history = model.fit(dataset, epochs=100, callbacks=[lr_schedule], verbose=0)
    return history


def plot_learning_rate(history):
    """plot loss vs learning rate per epoch"""
    lrs = 1e-8 * (10 ** (np.arange(100) / 20))
    plt.semilogx(lrs, history.history["loss"])
    plt.axis([1e-8, 1e-3, 0, 300])
    plt.show()


def set_optimal_learning_rate():
    """set optimal learning rate to be used in final model"""
    x_train, x_valid, time_train, time_valid = create_dataset()
    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    model = create_model()
    history = learning_rate_scheduler(model, dataset)
    plot_learning_rate(history)


def forecast(model, dataset, series):
    """predict output for validation data"""
    optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
    model.compile(loss="mse", optimizer=optimizer)
    history = model.fit(dataset, epochs=500, verbose=0)

    forecast = []
    for time in range(len(series)-window_size):
        forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    forecast = forecast[split_size-window_size:]
    results = np.array(forecast)[:, 0, 0]
    return results


def rainfall_prediction():
    """predict future rainfall and generate mean absolute error"""
    series, time = csv_parser()
    x_train, x_valid, time_train, time_valid = split_train_valid(split_size, series, time)

    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    model = create_model()

    results = forecast(model, dataset, series)
    errors = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
    print(errors)


if __name__ == '__main__':
    # set_optimal_learning_rate()
    rainfall_prediction()
