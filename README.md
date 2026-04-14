# House Price ML (Beginner Project)

This is a beginner-friendly machine learning project that predicts house prices using **Linear Regression**.

The project uses:
- **pandas** for loading and working with data
- **scikit-learn** for training and evaluating the model

---

## Dataset

Place a CSV file named `house_prices.csv` in the project root folder.

Expected columns:
- `SquareFeet`
- `Bedrooms`
- `Age`
- `Price`

Example:

```csv
SquareFeet,Bedrooms,Age,Price
1500,3,10,250000
1800,3,5,320000
2200,4,8,410000
```

---

## Script Included

- `house_price_linear_regression.py`

The script does the following:
1. Loads the dataset
2. Separates features (`X`) and target (`y`)
3. Splits data into training/testing sets (80/20)
4. Trains a Linear Regression model
5. Makes predictions on test data
6. Evaluates with MAE, MSE, and R²
7. Prints actual vs predicted values
8. Predicts the price of a new house

---

## Setup

Install dependencies:

```bash
pip install pandas scikit-learn
```

---

## Run the Project

```bash
python house_price_linear_regression.py
```

---

## Learning Goal

This project is designed to help beginners understand a complete machine learning workflow with simple, readable Python code.
