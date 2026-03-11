# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'Size': [1000, 1500, 2000, 2500, 3000],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Age': [10, 5, 3, 2, 1],
    'Price': [200000, 300000, 400000, 500000, 600000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Independent variables (multiple variables)
X = df[['Size', 'Bedrooms', 'Age']]

# Dependent variable
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Model evaluation
print("Predicted Values:", y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
