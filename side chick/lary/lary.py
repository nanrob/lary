import numpy as np  # import NumPy and call it np for array math

# Training data: x is the input, y is the expected output
x = np.array([[1.0], [2.0], [3.0], [4.0]])  # four input samples, each with one feature
y = np.array([[2.0], [4.0], [6.0], [8.0]])  # matching target values for each input

# Neural network hyperparameters
hidden_neurons = 16  # number of neurons in the hidden layer
input_neurons = 1  # single input value per sample
output_neurons = 1  # single output value per sample
learning_rate = 0.01  # step size for gradient descent updates
epochs = 4000  # how many times to train over the full dataset

# Random number generator for weight initialization
rng = np.random.default_rng(42)
W1 = rng.normal(scale=0.1, size=(input_neurons, hidden_neurons))  # weights from input to hidden layer
b1 = np.zeros((1, hidden_neurons))  # biases for the hidden layer
W2 = rng.normal(scale=0.1, size=(hidden_neurons, output_neurons))  # weights from hidden to output layer
b2 = np.zeros((1, output_neurons))  # biases for the output layer

# Activation function for hidden neurons
def relu(z):
    return np.maximum(0, z)  # ReLU: returns zero for negative inputs, otherwise returns input value

# Derivative of ReLU needed for backpropagation
def relu_derivative(z):
    return (z > 0).astype(float)  # 1 where z>0, else 0

for epoch in range(epochs):  # training loop
    # Forward pass through the network
    hidden_input = x.dot(W1) + b1  # input to hidden layer neurons
    hidden_output = relu(hidden_input)  # apply activation to hidden layer
    predictions = hidden_output.dot(W2) + b2  # output layer predictions

    # Compute the mean squared error loss
    error = predictions - y  # difference between prediction and truth
    loss = np.mean(error**2)  # average squared error

    # Backpropagation
    d_predictions = 2 * error / x.shape[0]  # derivative of MSE loss w.r.t. predictions
    dW2 = hidden_output.T.dot(d_predictions)  # gradient for W2
    db2 = np.sum(d_predictions, axis=0, keepdims=True)  # gradient for b2

    d_hidden = d_predictions.dot(W2.T) * relu_derivative(hidden_input)  # gradient through hidden layer
    dW1 = x.T.dot(d_hidden)  # gradient for W1
    db1 = np.sum(d_hidden, axis=0, keepdims=True)  # gradient for b1

    # Update weights and biases using gradient descent
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, loss: {loss:.6f}")  # print progress every 500 epochs

# Predict for a new input after training
x_test = np.array([[10.0]])  # new input value to predict
hidden_test = relu(x_test.dot(W1) + b1)  # forward pass through hidden layer
prediction = hidden_test.dot(W2) + b2  # final output prediction
print(f"Prediction for 10: {prediction[0, 0]:.4f}")  # print the predicted result