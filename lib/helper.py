from settings import file_path, rainfall_column

import matplotlib.pyplot as plt
import numpy as np
import csv


def csv_parser():
    """parse csv file data and generate rainfall data with timestep"""
    time_step = []
    rainfall = []

    with open(file_path) as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        counter = 1

        for row in reader:
            rainfall.append(float(row[rainfall_column]))
            time_step.append(counter)
            counter = counter + 1

        series = np.array(rainfall)
        time = np.array(time_step)

        return series, time


def plot_series(series, time, format="-", start=0, end=None):
    """plot series against time for visualization"""
    plt.figure(figsize=(10, 6))
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
