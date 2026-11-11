import numpy as np

x = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([[2.0], [4.0], [6.0], [8.0]])

hidden_neurons = 16
input_neurons = 1
output_neurons = 1
learning_rate = 0.01
epochs = 2000

rng = np.random.default_rng(42)
W1 = rng.normal(scale=0.1, size=(input_neurons, hidden_neurons))
b1 = np.zeros((1, hidden_neurons))
W2 = rng.normal(scale=0.1, size=(hidden_neurons, output_neurons))
b2 = np.zeros((1, output_neurons))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

for epoch in range(epochs):
    hidden_input = x.dot(W1) + b1
    hidden_output = relu(hidden_input)
    predictions = hidden_output.dot(W2) + b2

    error = predictions - y
    loss = np.mean(error**2)

    d_predictions = 2 * error / x.shape[0]
    dW2 = hidden_output.T.dot(d_predictions)
    db2 = np.sum(d_predictions, axis=0, keepdims=True)

    d_hidden = d_predictions.dot(W2.T) * relu_derivative(hidden_input)
    dW1 = x.T.dot(d_hidden)
    db1 = np.sum(d_hidden, axis=0, keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss: {loss:.6f}")

x_test = np.array([[233.0]])
hidden_test = relu(x_test.dot(W1) + b1)
prediction = hidden_test.dot(W2) + b2
print(f"Prediction for 10: {prediction[0, 0]:.4f}")