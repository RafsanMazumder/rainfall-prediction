## Features and Labels

The features are the descriptive attributes, and the label is what we're attempting to predict or forecast. In this project, 
the features are a set of rainfall data, with our label being the next rainfall value. So for example, if we take our time 
series data, say, 30 days at a time, we'll use 30 values as the feature and the next value is the label. Then over time, we'll
train a neural network to match the 30 features to the single label.


## Window size, Shuffle and Batch

The number of elements in a feature set is the window size. Shuffle helps us to rearrange the data so as not to accidentally introduce a sequence bias. Multiple runs will show the data in different arrangements because it gets shuffled randomly. By setting a batch size of (let's say for example) two, our data gets batched into two x's and two y's at a time.

## Windowed_dataset example

```python
x = [[4 5 6 7] [1 2 3 4]]
y = [[8] [5]]

x = [[2 3 4 5] [3 4 5 6]]
y = [[6] [7]]

x = [[5 6 7 8] [0 1 2 3]]
y = [[9] [4]]
```
