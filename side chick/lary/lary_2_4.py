import numpy as np

# Data setup
x = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [3.0, 6.0, 9.0, 12.0], [4.0, 8.0, 12.0, 16.0]])
y = np.array([[2.0, 4.0, 6.0, 8.0], [4.0, 8.0, 12.0, 16.0], [6.0, 12.0, 18.0, 24.0], [8.0, 16.0, 24.0, 32.0]])

# Layer configuration
dims = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,]
learning_rate = 0.01 # Lowered for stability in deep networks
epochs = 10000

# 1. Initialize all 11 Layers
rng = np.random.default_rng(42)
weights = [rng.normal(scale=0.1, size=(dims[i], dims[i+1])) for i in range(len(dims)-1)]
biases = [np.zeros((1, dims[i+1])) for i in range(len(dims)-1)]

def relu(z): return np.maximum(0, z)
def relu_derivative(z): return (z > 0).astype(float)

for epoch in range(epochs):
    # 2. Forward Pass
    activations = [x]
    linear_inputs = []
    
    for i in range(len(weights)):
        z = activations[-1].dot(weights[i]) + biases[i]
        linear_inputs.append(z)
        # Apply ReLU to all hidden layers, but not the final output layer
        a = relu(z) if i < len(weights) - 1 else z
        activations.append(a)

    predictions = activations[-1]
    error = predictions - y
    loss = np.mean(error**2)

    # 3. Backward Pass (Backpropagation)
    # Output layer gradient
    delta = 2 * error / x.shape[0]
    
    for i in reversed(range(len(weights))):
        dW = activations[i].T.dot(delta)
        db = np.sum(delta, axis=0, keepdims=True)
        
        # Calculate delta for the previous layer
        if i > 0:
            delta = delta.dot(weights[i].T) * relu_derivative(linear_inputs[i-1])
        
        # 4. Parameter Updates
        weights[i] -= learning_rate * dW
        biases[i] -= learning_rate * db

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, loss: {loss:.6f}")

print(f"Final Prediction for Input:\n{activations[-1]}")
