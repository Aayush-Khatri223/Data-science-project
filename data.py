import numpy as np
import pandas as pd

# Data from the image
data = {
    'x': [1, 2,3,4,5],
    'y': [3,4,5,6,7]
}

df = pd.DataFrame(data)
print("Input Data:")
print(df)
print()

# Extract x and y as numpy arrays
x = np.array(df['x'])
y = np.array(df['y'])

# Number of samples
m = len(x)

# Initialize parameters
theta0 = 2      # intercept (fixed as per y = 2 + 0.8*x)
theta1 = 0.8    # slope (initial value)
alpha = 0.01    # learning rate

# Number of iterations
iterations = 1000

print(f"Initial Parameters: theta0 = {theta0}, theta1 = {theta1}")
print(f"Learning Rate (alpha) = {alpha}")
print(f"Initial prediction: y = {theta0} + {theta1}*x")
print()

# Gradient Descent
print("Gradient Descent Iterations:")
print("-" * 50)

for i in range(iterations):
    # Predicted y values: y_pred = theta0 + theta1 * x
    y_pred = theta0 + theta1 * x
    
    # Calculate error
    error = y_pred - y
    
    # Calculate cost (Mean Squared Error)
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    
    # Calculate gradients
    gradient_theta0 = (1 / m) * np.sum(error)
    gradient_theta1 = (1 / m) * np.sum(error * x)
    
    # Update parameters using gradient descent (theta0 is kept constant at 2)
    theta1 = theta1 - alpha * gradient_theta1
    
    # Print progress every 100 iterations
    if i % 100 == 0 or i == iterations - 1:
        print(f"Iteration {i:4d}: theta0 = {theta0:.6f}, theta1 = {theta1:.6f}, Cost = {cost:.6f}")

print("-" * 50)
print()

# Final Results
print("Final Results:")
print(f"Optimized theta0 (intercept) = {theta0:.6f}")
print(f"Optimized theta1 (slope) = {theta1:.6f}")
print(f"Best fit equation: y = {theta0:.4f} + {theta1:.4f}*x")
print()

# Final predictions
y_final = theta0 + theta1 * x

# Create results dataframe
results_df = pd.DataFrame({
    'x': x,
    'y_actual': y,
    'y_predicted': np.round(y_final, 4)
})

print("Predictions vs Actual:")
print(results_df)
print()

# Calculate final MSE
final_mse = np.mean((y_final - y) ** 2)
print(f"Final Mean Squared Error: {final_mse:.6f}")
