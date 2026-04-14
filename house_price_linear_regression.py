"""
Beginner-friendly house price prediction example using Linear Regression.

This script expects a CSV file named "house_prices.csv" in the same folder.
Required columns in the CSV:
- SquareFeet
- Bedrooms
- Age
- Price
"""

# Import pandas for loading and working with table-like data.
import pandas as pd

# Import train_test_split to split our data into training and testing parts.
from sklearn.model_selection import train_test_split

# Import LinearRegression, a simple machine learning model for predicting numbers.
from sklearn.linear_model import LinearRegression

# Import evaluation metrics so we can measure how good our model is.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# STEP 1: Load the dataset from a CSV file into a pandas DataFrame.
# A DataFrame is like a spreadsheet in Python.
data = pd.read_csv("house_prices.csv")

# Show the first few rows so beginners can quickly verify the file loaded correctly.
print("First 5 rows of the dataset:")
print(data.head())
print("\n")


# STEP 2: Separate features (X) and target (y).
# Features are the input columns the model uses to learn patterns.
# Target is what we want to predict (house price).
X = data[["SquareFeet", "Bedrooms", "Age"]]
y = data["Price"]


# STEP 3: Split the data into training and testing sets.
# - 80% of the data is used to train the model.
# - 20% of the data is used to test how well the model performs on unseen data.
# random_state=42 keeps the split the same each time for reproducible results.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# STEP 4: Create and train a Linear Regression model.
# The model learns the relationship between features and price from training data.
model = LinearRegression()
model.fit(X_train, y_train)


# STEP 5: Make predictions using the test features.
# These predictions are the model's estimated prices for the test houses.
y_pred = model.predict(X_test)


# STEP 6: Evaluate model performance with MAE, MSE, and R^2.
# MAE: Average absolute difference between actual and predicted prices.
# MSE: Average squared difference (penalizes larger errors more).
# R^2: How much of the price variation is explained by the model (closer to 1 is better).
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R^2: {r2:.4f}")
print("\n")


# STEP 7: Print actual vs predicted values so beginners can compare results directly.
comparison = pd.DataFrame(
    {
        "Actual Price": y_test.values,
        "Predicted Price": y_pred,
    }
)

print("Actual vs Predicted Prices (test set):")
print(comparison)
print("\n")


# STEP 8: Predict the price of a new house.
# We create a one-row DataFrame with the same feature columns used during training.
new_house = pd.DataFrame(
    {
        "SquareFeet": [2200],
        "Bedrooms": [4],
        "Age": [8],
    }
)

# Ask the trained model to predict the new house price.
new_price_prediction = model.predict(new_house)[0]

print("Predicted price for a new house (2200 sq ft, 4 bedrooms, 8 years old):")
print(f"${new_price_prediction:,.2f}")
