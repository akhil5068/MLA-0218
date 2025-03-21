import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Generate synthetic data (Quadratic Relationship)
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2 * X**2 + 3 * X + np.random.randn(50, 1) * 10  # Quadratic relationship with noise

# Step 2: Train a Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)

# Step 3: Train a Polynomial Regression Model (Degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)  # Transform features to polynomial features
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Step 4: Visualization
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred_linear, color="red", linestyle="dashed", label="Linear Regression")
plt.plot(X, y_pred_poly, color="green", label="Polynomial Regression (Degree 2)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Comparison of Linear vs Polynomial Regression")
plt.show()

# Step 5: Print Model Coefficients
print("Linear Regression Coefficients:", lin_reg.coef_, "Intercept:", lin_reg.intercept_)
print("Polynomial Regression Coefficients:", poly_reg.coef_, "Intercept:", poly_reg.intercept_)