# lary_2_5.py Explained

This README explains the `lary_2_5.py` script in detail. The script implements a small neural network from scratch using NumPy, trains it on a simple `y = 2x` regression task, and tests the trained network on new inputs.

## File purpose

`lary_2_5.py` is designed to teach the basics of:

- creating a multilayer perceptron (MLP)
- forward propagation
- mean squared error loss
- backpropagation
- gradient descent updates
- prediction on test data
- adaptive neuron capacity: pruning and growing hidden units automatically
- improved training in `lary_2_6.py`: more data, momentum, validation, and weight decay

## Structure overview

The script is organized into three main sections:

1. `MLPRegressor` class definition
2. `main()` function to train and test the model
3. `if __name__ == "__main__": main()` entry point

## Detailed explanation

### 1. Imports

```python
import numpy as np
```

- Imports NumPy, the numerical Python library.
- NumPy provides arrays and matrix math used throughout the network.

### 2. Class definition

```python
class MLPRegressor:
    """A small feedforward neural network for regression.

    This class implements a simple multilayer perceptron using NumPy only.
    It supports ReLU activation in hidden layers and a linear output layer.
    """
```

- Defines a Python class named `MLPRegressor`.
- The docstring explains the model's purpose.

#### `__init__`

```python
    def __init__(self, layer_sizes, learning_rate=1e-3, seed=42, max_hidden=32):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.max_hidden = max_hidden
        self.rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        self._initialize_parameters()
```

- `layer_sizes` is a list describing the number of neurons in each layer.
- `learning_rate` controls how large each update step is.
- `seed` ensures the same random initialization each run.
- `max_hidden` is the maximum number of hidden neurons allowed in a layer when the network grows.
- `rng` is the random number generator.
- `weights` and `biases` are initialized as empty lists.
- `_initialize_parameters()` fills them with actual arrays.

#### `_initialize_parameters`

```python
    def _initialize_parameters(self):
        self.weights = [
            self.rng.normal(scale=0.01, size=(self.layer_sizes[i], self.layer_sizes[i + 1]))
            for i in range(len(self.layer_sizes) - 1)
        ]
        self.biases = [np.zeros((1, self.layer_sizes[i + 1])) for i in range(len(self.layer_sizes) - 1)]
```

- Creates one weight matrix per layer connection.
- Each weight matrix has shape `(input_dim, output_dim)`.
- We use small random values centered around zero.
- Biases are initialized to zeros with one row.

#### Activation helpers

```python
    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)
```

- `relu` applies the Rectified Linear Unit activation function.
- `relu_derivative` computes the derivative, which is needed for backpropagation.
- `@staticmethod` means these functions do not use the class instance state.

#### `forward`

```python
    def forward(self, x):
        activations = [x]
        preactivations = []

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            preactivations.append(z)
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = z
            activations.append(a)

        return activations, preactivations
```

- `activations` starts with the input data.
- `preactivations` stores each layer's raw `z = xW + b` values.
- For each layer:
  - compute `z` using matrix multiplication and add bias.
  - store `z` for later gradient computation.
  - apply ReLU only on hidden layers.
  - use a linear activation on the final output layer.
- Returns the full activation history and preactivation history.

#### `compute_loss`

```python
    def compute_loss(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)
```

- Computes mean squared error (MSE) loss.
- `predictions - targets` is the error vector.
- Squaring and averaging produces a single scalar loss.

#### `backward`

```python
    def backward(self, activations, preactivations, y_true):
        grads_w = []
        grads_b = []

        predictions = activations[-1]
        error = predictions - y_true

        delta = 2.0 * error / y_true.shape[0]

        for i in reversed(range(len(self.weights))):
            dw = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            grads_w.insert(0, np.clip(dw, -1.0, 1.0))
            grads_b.insert(0, np.clip(db, -1.0, 1.0))

            if i > 0:
                delta = delta @ self.weights[i].T
                delta *= self.relu_derivative(preactivations[i - 1])
                delta = np.clip(delta, -1.0, 1.0)

        return grads_w, grads_b
```

- Builds gradients for all weights and biases.
- `error` is the difference between predictions and true labels.
- `delta` starts as the derivative of the MSE loss.
- For each layer in reverse order:
  - compute `dw` using the activation from the previous layer.
  - compute `db` by summing across the batch.
  - clip gradients to keep them from becoming too large.
- If not the first layer, propagate the error backward through the weights.
- Multiply by the derivative of ReLU for the hidden layer.
- Returns lists of gradients in the same order as weights and biases.

#### `update_parameters`

```python
    def update_parameters(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]
```

- Applies gradient descent updates.
- Each weight and bias is moved in the opposite direction of its gradient.
- The step size is controlled by `learning_rate`.

#### `fit`

```python
    def fit(
        self,
        x,
        y,
        epochs=10000,
        print_every=500,
        tol=1e-8,
        auto_adjust=False,
        adjust_every=1000,
        grow_patience=3,
        prune_threshold=1e-3,
    ):
        history = []

        for epoch in range(epochs):
            activations, preactivations = self.forward(x)
            loss = self.compute_loss(activations[-1], y)
            history.append(loss)

            grads_w, grads_b = self.backward(activations, preactivations, y)
            self.update_parameters(grads_w, grads_b)

            if loss < best_loss - 1e-12:
                best_loss = loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if auto_adjust and epoch > 0 and epoch % adjust_every == 0:
                if no_improve_count >= grow_patience:
                    self.grow_neurons()
                    no_improve_count = 0
                else:
                    self.prune_neurons(threshold=prune_threshold)

            if epoch % print_every == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:5d} | loss = {loss:.8f}")

            if loss < tol:
                print(f"Early stopping at epoch {epoch}: loss {loss:.8f}")
                break

        return history
```

- Trains the model for a fixed number of epochs.
- Runs a forward pass, computes loss, backpropagates gradients, and updates parameters.
- `auto_adjust=True` allows the network to add or remove hidden neurons automatically.
- `adjust_every` sets how often capacity is reevaluated.
- `grow_patience` controls how many checks without improvement trigger growth.
- `prune_threshold` sets how small a neuron's contribution must be before it is removed.
- `history` collects loss values per epoch.
- Prints progress every `print_every` epochs.
- Stops early if loss falls below `tol`.

### lary_2_6 improvements

`lary_2_6.py` adds a stronger training setup on top of the original toy model:

- uses a larger synthetic training set so the medium-sized model can actually learn
- splits data into training and validation sets
- uses a moderate medium-sized architecture like `[4, 64, 64, 32, 4]`
- adds momentum-based updates to smooth optimization
- applies a small weight decay penalty to discourage overly large weights
- supports early stopping based on validation loss
- reduces the model size compared to the first version while keeping enough capacity for generalization

### How to run `lary_2_6.py`

In the terminal, run:

```bash
py "c:\Users\rober\OneDrive\Desktop\New folder (2)\side chick\lary\lary_2_6.py"
```

This script now:

- generates 1000 synthetic training examples
- trains a medium-sized model with momentum and weight decay
- uses 20% of the data for validation
- stops early when validation loss stops improving

### Comparing `lary_2_5.py` and `lary_2_6.py`

- `lary_2_5.py` is the smaller educational version with a tiny dataset and simple training.
- `lary_2_6.py` is the medium-sized version with a larger dataset, validation monitoring, and better optimization.

#### `predict`

```python
    def predict(self, x):
        activations, _ = self.forward(x)
        return activations[-1]
```

- Computes network outputs for new input data.
- Uses the `forward` method and returns the final layer activation.

### 3. Main training and testing block

```python
def main():
    x_train = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [3.0, 6.0, 9.0, 12.0],
        [4.0, 8.0, 12.0, 16.0],
    ])
    y_train = x_train * 2.0
```

- Creates the simple training dataset.
- Each target is exactly two times the input values.

```python
    network = MLPRegressor(layer_sizes=[4, 16, 16, 8, 4], learning_rate=1e-3)
    history = network.fit(
        x_train,
        y_train,
        epochs=12000,
        print_every=1000,
        auto_adjust=True,
        adjust_every=2000,
        grow_patience=4,
        prune_threshold=1e-3,
    )
```

- Builds the neural network with 4 input features, two hidden layers of 16 units,
  a smaller hidden layer with 8 units, and 4 outputs.
- Trains for up to 12,000 epochs, printing every 1,000 epochs.

```python
    x_test = np.array([
        [5.0, 10.0, 15.0, 20.0],
        [7.0, 14.0, 21.0, 28.0],
    ])
    y_test = x_test * 2.0
    predictions = network.predict(x_test)
```

- Defines new test inputs that follow the same `y = 2x` rule.
- Uses the trained network to predict outputs.

```python
    print("\n--- Final Predictions ---")
    for i in range(len(x_test)):
        print(f"Input:      {x_test[i]}")
        print(f"Expected:   {y_test[i]}")
        print(f"Predicted:  {predictions[i]}\n")

    final_loss = network.compute_loss(predictions, y_test)
    print(f"Test MSE: {final_loss:.10f}")
```

- Prints each test input, expected output, and model prediction.
- Computes test mean squared error to show how close predictions are.

### 4. Script entry point

```python
if __name__ == "__main__":
    main()
```

- Ensures `main()` runs only when the file is executed directly.
- Prevents training from running if the file is imported as a module.

## How to use

1. Open `lary_2_5.py` in Python.
2. Run the script.
3. Observe training loss and final predictions.

## Notes

- This implementation is educational and not optimized for large datasets.
- It uses a fixed learning rate and no regularization.
- For real problems, consider frameworks such as TensorFlow or PyTorch.
