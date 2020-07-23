import tensorflow as tf


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """prepare a time series for machine learning by windowing the data"""
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def split_train_valid(split_time, series, time):
    """split the data series into training and validation portions"""
    x_train = series[:split_time]
    x_valid = series[split_time:]
    time_train = time[:split_time]
    time_valid = time[split_time:]
    return x_train, x_valid, time_train, time_valid



