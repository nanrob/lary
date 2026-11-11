import numpy as np

x = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 4.0, 6.0, 8.0],
    [3.0, 6.0, 9.0, 12.0],
    [4.0, 8.0, 12.0, 16.0]
])
y = np.array([
    [2.0, 4.0, 6.0, 8.0],
    [4.0, 8.0, 12.0, 16.0],
    [6.0, 12.0, 18.0, 24.0],
    [8.0, 16.0, 24.0, 32.0]
])

hidden_neurons1 = 8
hidden_neurons2 = 16                     
hidden_neurons3 = 32
hidden_neurons4 = 64
hidden_neurons5 = 128
hidden_neurons6 = 256
hidden_neurons7 = 512
hidden_neurons8 = 1024
hidden_neurons9 = 2048
hidden_neurons10 = 4096

input_neurons = 4
output_neurons = 4
learning_rate = 0.01
epochs = 10000

rng = np.random.default_rng(42)
W1 = rng.normal(scale=0.1, size=(input_neurons, hidden_neurons1))
b1 = np.zeros((1, hidden_neurons1))
W2 = rng.normal(scale=0.1, size=(hidden_neurons1, hidden_neurons2))
b2 = np.zeros((1, hidden_neurons2))
W3 = rng.normal(scale=0.1, size=(hidden_neurons2, output_neurons))
b3 = np.zeros((1, output_neurons))
b2 = np.zeros((1, hidden_neurons2))



def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

for epoch in range(epochs):
    hidden1_input = x.dot(W1) + b1
    hidden1_output = relu(hidden1_input)
    hidden2_input = hidden1_output.dot(W2) + b2
    hidden2_output = relu(hidden2_input)
    predictions = hidden2_output.dot(W3) + b3

    error = predictions - y
    loss = np.mean(error**2)

    d_predictions = 2 * error / x.shape[0]
    dW3 = hidden2_output.T.dot(d_predictions)
    db3 = np.sum(d_predictions, axis=0, keepdims=True)

    d_hidden2 = d_predictions.dot(W3.T) * relu_derivative(hidden2_input)
    dW2 = hidden1_output.T.dot(d_hidden2)
    db2 = np.sum(d_hidden2, axis=0, keepdims=True)

    d_hidden1 = d_hidden2.dot(W2.T) * relu_derivative(hidden1_input)
    dW1 = x.T.dot(d_hidden1)
    db1 = np.sum(d_hidden1, axis=0, keepdims=True)

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, loss: {loss:.6f}")

x_test = np.array([[5.0, 10.0, 15.0, 20.0]])
hidden1_test = relu(x_test.dot(W1) + b1)
hidden2_test = relu(hidden1_test.dot(W2) + b2)
prediction = hidden2_test.dot(W3) + b3
print(f"Prediction: {prediction[0]}")