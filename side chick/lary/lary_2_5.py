import numpy as np


class MLPRegressor:
    """A small feedforward neural network for regression.

    This class implements a simple multilayer perceptron using NumPy only.
    It supports ReLU activation in hidden layers and a linear output layer.
    """

    def __init__(self, layer_sizes, learning_rate=1e-3, seed=42, max_hidden=250):
        # layer_sizes defines the number of units in each layer, e.g. [4, 16, 16, 8, 4]
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.max_hidden = max_hidden
        self.rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        self._initialize_parameters()

    def _initialize_parameters(self):
        # Initialize the weights with small random values and biases with zeros.
        # Weight matrices connect each layer to the next layer.
        self.weights = [
            self.rng.normal(scale=0.01, size=(self.layer_sizes[i], self.layer_sizes[i + 1]))
            for i in range(len(self.layer_sizes) - 1)
        ]
        self.biases = [np.zeros((1, self.layer_sizes[i + 1])) for i in range(len(self.layer_sizes) - 1)]

    @staticmethod
    def relu(z):
        # ReLU activation function: max(0, z)
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z):
        # Derivative of ReLU used during backpropagation.
        return (z > 0).astype(float)

    def forward(self, x):
        """Compute the forward pass.

        Returns all layer activations and pre-activation values (z) needed for backprop.
        """
        activations = [x]
        preactivations = []

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Compute the affine transform for this layer.
            z = activations[-1] @ w + b
            preactivations.append(z)

            # Apply ReLU to hidden layers only; the last layer is linear for regression.
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = z
            activations.append(a)

        return activations, preactivations

    def compute_loss(self, predictions, targets):
        # Mean squared error loss for regression.
        return np.mean((predictions - targets) ** 2)

    def backward(self, activations, preactivations, y_true):
        """Compute gradients using backpropagation."""
        grads_w = []
        grads_b = []

        predictions = activations[-1]
        error = predictions - y_true

        # Derivative of MSE loss with respect to predictions.
        delta = 2.0 * error / y_true.shape[0]

        for i in reversed(range(len(self.weights))):
            # Compute gradients for weights and biases at layer i.
            dw = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            # Clip gradients for numerical stability.
            grads_w.insert(0, np.clip(dw, -1.0, 1.0))
            grads_b.insert(0, np.clip(db, -1.0, 1.0))

            if i > 0:
                # Propagate the error back through the previous layer.
                delta = delta @ self.weights[i].T
                delta *= self.relu_derivative(preactivations[i - 1])
                delta = np.clip(delta, -1.0, 1.0)

        return grads_w, grads_b

    def update_parameters(self, grads_w, grads_b):
        # Apply gradient descent updates to weights and biases.
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def prune_neurons(self, min_neurons=2, threshold=1e-3):
        """Remove hidden neurons that have very little impact."""
        for layer_index in range(len(self.weights) - 1):
            if self.weights[layer_index].shape[1] <= min_neurons:
                continue

            incoming_norm = np.linalg.norm(self.weights[layer_index], axis=0)
            outgoing_norm = np.linalg.norm(self.weights[layer_index + 1], axis=1)
            importance = incoming_norm + outgoing_norm

            keep_mask = importance >= threshold
            if np.sum(keep_mask) < min_neurons:
                # Keep the strongest neurons if threshold removes too many.
                keep_indices = np.argsort(-importance)[:min_neurons]
                keep_mask = np.zeros_like(importance, dtype=bool)
                keep_mask[keep_indices] = True

            if np.sum(keep_mask) == importance.size:
                continue

            self.weights[layer_index] = self.weights[layer_index][:, keep_mask]
            self.biases[layer_index] = self.biases[layer_index][:, keep_mask]
            self.weights[layer_index + 1] = self.weights[layer_index + 1][keep_mask, :]

            print(f"Pruned layer {layer_index} to {self.weights[layer_index].shape[1]} neurons.")

    def grow_neurons(self, grow_amount=1):
        """Add a small number of neurons to hidden layers if the network is stuck."""
        for layer_index in range(len(self.weights) - 1):
            current_neurons = self.weights[layer_index].shape[1]
            if current_neurons >= self.max_hidden:
                continue

            add = min(grow_amount, self.max_hidden - current_neurons)
            new_weights = self.rng.normal(
                scale=0.01, size=(self.weights[layer_index].shape[0], add)
            )
            self.weights[layer_index] = np.concatenate([self.weights[layer_index], new_weights], axis=1)
            self.biases[layer_index] = np.concatenate(
                [self.biases[layer_index], np.zeros((1, add))], axis=1
            )
            self.weights[layer_index + 1] = np.concatenate(
                [
                    self.weights[layer_index + 1],
                    self.rng.normal(scale=0.01, size=(add, self.weights[layer_index + 1].shape[1])),
                ],
                axis=0,
            )
            print(f"Grew layer {layer_index} to {self.weights[layer_index].shape[1]} neurons.")

    def fit(
        self,
        x,
        y,
        epochs=100,
        print_every=50,
        tol=1e-8,
        auto_adjust=False,
        adjust_every=100,
        grow_patience=3,
        prune_threshold=1e-3,
    ):
        """Train the network on the given data."""
        history = []
        best_loss = float("inf")
        no_improve_count = 0

        for epoch in range(epochs):
            activations, preactivations = self.forward(x)
            loss = self.compute_loss(activations[-1], y)
            history.append(loss)

            grads_w, grads_b = self.backward(activations, preactivations, y)
            self.update_parameters(grads_w, grads_b)

            if loss < best_loss - 1e-12:
                # Training is still improving: reset the counter.
                best_loss = loss
                no_improve_count = 0
            else:
                # No improvement this epoch: increase the patience counter.
                no_improve_count += 1

            # Every `adjust_every` epochs, decide whether to grow or prune.
            # - Grow if the network has been stuck for `grow_patience` checks.
            # - Otherwise, prune weak neurons to reduce wasted capacity.
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

    def predict(self, x):
        """Return predictions for new input data."""
        activations, _ = self.forward(x)
        return activations[-1]


def main():
    # Training data: simple examples where y = 2 * x.
    x_train = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [3.0, 6.0, 9.0, 12.0],
        [4.0, 8.0, 12.0, 16.0],
    ])
    y_train = x_train * 2.0

    # Create the model and train it with automatic capacity adjustment.
    network = MLPRegressor(layer_sizes=[4, 16, 16, 8, 4], learning_rate=1e-3)
    history = network.fit(
        x_train,
        y_train,
        epochs=50000,
        print_every=100,
        auto_adjust=True,
        adjust_every=100,
        grow_patience=4,
        prune_threshold=1e-3,
    )

    # Test the trained model on new inputs.
    x_test = np.array([
        [5.0, 10.0, 15.0, 20.0],
        [7.0, 14.0, 21.0, 28.0],
    ])
    y_test = x_test * 2.0
    predictions = network.predict(x_test)

    print("\n--- Final Predictions ---")
    for i in range(len(x_test)):
        print(f"Input:      {x_test[i]}")
        print(f"Expected:   {y_test[i]}")
        print(f"Predicted:  {predictions[i]}\n")

    final_loss = network.compute_loss(predictions, y_test)
    print(f"Test MSE: {final_loss:.10f}")


if __name__ == "__main__":
    main()
