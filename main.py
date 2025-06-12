import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Define feature names
temp = ["Age", "ShoulderToWaist", "WaistToKnee", "TotalHeight", "ShoulderWidth", "ArmLength", "Hips"]
tempWOAge = ["ShoulderToWaist", "WaistToKnee", "TotalHeight", "ShoulderWidth", "ArmLength", "Hips"]

# Load the dataset
df = pd.read_csv("Body_Measurements.csv")

# Select relevant columns and clean column names
df = df[temp]
df.columns = df.columns.str.strip() 

print("Sample ages and features:\n", df.head())

# Pearson correlation matrix
correlation_matrix = df.corr()
print("Pearson Correlation Matrix:\n", correlation_matrix)

# Plot correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Pearson Correlation Matrix")
plt.show()

# Split data into train and test sets
X = df[tempWOAge]
y = df["Age"]

# Split the data randomly using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Display model coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict values
y_pred = model.predict(X_test_scaled)

# Calculate MSE and R2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE (Mean Squared Error):", mse)
print("R² (R-squared):", r2)

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs Predicted Age")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()

# Plot residuals for model evaluation
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Age")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.show()
