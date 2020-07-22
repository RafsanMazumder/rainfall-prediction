# rainfall-prediction
rainfall-prediction is based on Neural Network in python 3.6.9, to predict precipitation in a river delta region like Bangladesh. Rainfall, considered as time series data, is used to develop, simulate, test and improve algorithms and the current model performs 5% better than statistical models.

## Usage main.py

```python
if __name__ == '__main__':
    set_optimal_learning_rate() # set the optimal learning rate by tuning our model
    rainfall_prediction() # generate prediction on the given dataset
```

## Parameters settings.py

```python
# Fetches parameters from environment file
file_path = os.getenv('file_path') # dataset file location
rainfall_column = int(os.getenv('rainfall_column')) # rainfall column index in dataset
split_size = int(os.getenv('split_size')) # test-train split point 
```
