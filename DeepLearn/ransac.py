import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data with outliers
np.random.seed(0)
x = np.linspace(0, 10, 100)  # Generate 100 evenly spaced x values between 0 and 10
y = 2 * x + 1 + np.random.normal(0, 1, 100)  # Generate corresponding y values with added noise
# Add some outliers
y[10] = 20  # Introduce an outlier
y[90] = 15  # Introduce another outlier

# Plot the data
plt.scatter(x, y, label='Data', color='blue')

# RANSAC Algorithm
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

# Create RANSAC and Linear Regression models
ransac = RANSACRegressor(LinearRegression(), max_trials=100, residual_threshold=2.0, random_state=0)

# Fit the RANSAC model to the data
ransac.fit(x.reshape(-1, 1), y)

# Mask inliers and outliers
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Plot inliers and outliers
plt.scatter(x[inlier_mask], y[inlier_mask], label='Inliers', color='green', marker='o')
plt.scatter(x[outlier_mask], y[outlier_mask], label='Outliers', color='red', marker='x')

# Plot the best fit line
x_range = np.array([0, 10])
y_range = ransac.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_range, color='orange', linewidth=2, label='RANSAC Line')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('RANSAC Algorithm Visualization')
plt.grid(True)
plt.show()
