## Features and Labels

The features are the descriptive attributes, and the label is what we're attempting to predict or forecast. In this project,  the features are a set of rainfall data, with our label being the next rainfall value. So for example, if we take our time series data, say, 30 days at a time, we'll use 30 values as the feature and the next value is the label. Then over time, we'll train a neural network to match the 30 features to the single label.


## Window size, Shuffle and Batch

The number of elements in a feature set is the window size. Shuffle helps us to rearrange the data so as not to accidentally  introduce a sequence bias. Multiple runs will show the data in different arrangements because it gets shuffled randomly. By setting a batch size of (let's say for example) two, our data gets batched into two x's and two y's at a time.

## Windowed Dataset example

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

## Learning Rate
The learning rate is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function. It is a hyper-parameter that controls how much we are adjusting the weights of our network with respect to the loss gradient. The lower the value, the slower we travel along the downward slope. While this might be a good idea (using a low learning rate) in terms of making sure that we do not miss any local minima, it could also mean that we’ll be taking a long time to converge. A learning rate schedule changes the learning rate during learning and is most often changed between epochs/iterations. 

## Layers in Neural Network

* Input Layer: Input variables, sometimes called the visible layer.
* Hidden Layers: Layers of nodes between the input and output layers. There may be one or more of these layers.
* Output Layer: A layer of nodes that produce the output variables.

Dense layer means, each neuron in a layer receives an input from all the neurons present in the previous layer—thus, they’re densely connected. 

In our case, we have used three layers of 10, 10, and 1 neurons and activated each layer using a relu (activation function).

## Activation Function

An activation function is a function that maps a node's inputs to its corresponding output. The activation function does some type of operation to transform the sum to a number that is often times between some lower limit and some upper limit. This transformation is often a non-linear transformation.

### Sigmoid Activation Function

Sigmoid takes in an input and does the following:

* For most negative inputs, sigmoid will transform the input to a number very close to 0.
* For most positive inputs, sigmoid will transform the input into a number very close to 1.
* For inputs relatively close to 0, sigmoid will transform the input into some number between 0 and 1.

### Relu Activation Function

ReLU, which is short for rectified linear unit, transforms the input to the maximum of either 0 or the input itself.

```python
def relu(x):
  if x <= 0:
    return 0
  else:
    return x
```

## Epoch

In terms of artificial neural networks, an epoch refers to one cycle through the full training dataset. Usually, training a neural network takes more than a few epochs.

## Picking Learning Rate

Learning rate scheduler changes the learning rates to a value based on the epoch number. From the loss (per epoch) vs learning rate (per epoch) grpah, We can try to pick the lowest point of the curve which is relatively stable. 









