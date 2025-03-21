import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset (Simulated dataset)
np.random.seed(42)
data_size = 1000
df = pd.DataFrame({
    'location': np.random.choice(['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Houston'], data_size),
    'size_sqft': np.random.randint(500, 5000, data_size),
    'bedrooms': np.random.randint(1, 6, data_size),
    'bathrooms': np.random.randint(1, 4, data_size),
    'year_built': np.random.randint(1900, 2023, data_size),
    'garage': np.random.randint(0, 2, data_size),  # 0: No garage, 1: Garage available
    'price': np.random.randint(100000, 2000000, data_size)  # Target variable
})

# Step 2: Data Preprocessing
# Convert categorical variable 'location' into numerical form using OneHotEncoding
df = pd.get_dummies(df, columns=['location'], drop_first=True)  # Avoid dummy variable trap

# Separate features and target
X = df.drop(columns=['price'])
y = df['price']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest Regressor Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)

# Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print Model Performance
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 6: Visualization of Actual vs Predicted Prices
plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()