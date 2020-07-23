## Features and Labels

The features are the descriptive attributes, and the label is what we're attempting to predict or forecast. In this project,  the features are a set of rainfall data, with our label being the next rainfall value. So for example, if we take our time series data, say, 30 days at a time, we'll use 30 values as the feature and the next value is the label. Then over time, we'll train a neural network to match the 30 features to the single label.


## Window size, Shuffle and Batch

The number of elements in a feature set is the window size. Shuffle helps us to rearrange the data so as not to accidentally  introduce a sequence bias. Multiple runs will show the data in different arrangements because it gets shuffled randomly. By setting a batch size of (let's say for example) two, our data gets batched into two x's and two y's at a time.

## Windowed dataset example

```python
x = [[4 5 6 7] [1 2 3 4]]
y = [[8] [5]]

x = [[2 3 4 5] [3 4 5 6]]
y = [[6] [7]]

x = [[5 6 7 8] [0 1 2 3]]
y = [[9] [4]]
```
## Gradient Descent
Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. To find  a local minimum of a function using gradient descent, we take steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point. But if we instead take steps proportional to the positive of the gradient, we approach a local maximum of that function; the procedure is then known as gradient ascent.

## Loss Function
Loss function a method of evaluating how well specific algorithm models the given data. If predictions deviates too much from actual results, loss function would cough up a very large number. 

Mean squared error (MSE) is the most commonly used loss function for regression. The loss is the mean overseen data of the squared differences between true and predicted values. We should use MSE, when we want large errors to be significantly (quadratically) more penalized than small ones.

## Neural Network Setup
We'll use the mean squared error as loss function by setting loss to MSE, and Gradient Descent as optimizer. 
