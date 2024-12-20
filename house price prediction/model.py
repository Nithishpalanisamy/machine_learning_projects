import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
data = pd.read_csv('HousingData.csv')

# Check for missing values
print("Missing values in dataset:\n", data.isnull().sum())

# Handle missing values
# Option 1: Fill missing values with the column mean (imputation)
data.fillna(data.mean(), inplace=True)

# Option 2: Drop rows with missing values
# data.dropna(inplace=True)

# Define features (X) and target (y)
X = data.drop('MEDV', axis=1)  # Assuming 'MEDV' is the target column
y = data['MEDV']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))

# Save model
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as 'house_price_model.pkl'")
