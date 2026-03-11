import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)
y = np.array([20, 40, 60, 80, 100])
model = LinearRegression()
model.fit(X, y)
predicted = model.predict([[12]])
print("Predicted Marks for 12 study hours:", predicted[0])
