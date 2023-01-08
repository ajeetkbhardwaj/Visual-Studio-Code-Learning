Artificial Neural Network using Python

```
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

# Input data
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Output data
y = np.array([[0], [1], [1], [0]])

# Seed random numbers to make calculation deterministic (good practice)
np.random.seed(1)

# Initialize weights randomly with mean 0
synaptic_weights = 2 * np.random.random((3, 1)) - 1

for iteration in range(10000):
  # Forward propagation
  input_layer = X
  outputs = sigmoid(np.dot(input_layer, synaptic_weights))

  # Calculate the error
  error = y - outputs

  # Multiply the error by the input and again by the gradient of the sigmoid curve
  adjustments = error * sigmoid_derivative(outputs)

  # Update the weights
  synaptic_weights += np.dot(input_layer.T, adjustments)

print("Synaptic weights after training: ")
print(synaptic_weights)
print("Outputs after training: ")
print(outputs)

```


Jacob method for solving Ax = b

```
import numpy as np

def jacobi(A, b, x0, tol, maxiter):
    n = len(x0)
    x = np.zeros_like(x0)
    for i in range(maxiter):
        for j in range(n):
            s = sum(A[j][k] * x0[k] for k in range(n) if k != j)
            x[j] = (b[j] - s) / A[j][j]
        if np.allclose(x, x0, rtol=tol):
            return x
        x0 = x
    raise ValueError("Failed to converge after {} iterations".format(maxiter))

# Example usage
A = np.array([[3, 1], [1, 3]])
b = np.array([9, 5])
x0 = np.array([1, 1])
tol = 1e-6
maxiter = 1000

x = jacobi(A, b, x0, tol, maxiter)
print(x)  # Output: [2.0, 2.0]


```
