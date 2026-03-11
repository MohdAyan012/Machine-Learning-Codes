#House Price Prediction with Graph (Linear Regression)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Linear Regression
#Dataset (Size in sq.ft vs Price in Lakhs)
X= np.array([500, 800, 1000, 1200, 1500, 1800, 2000]).reshape(-1, 1) y= np.array([10, 18, 25, 30, 40, 50, 55])
model Linear Regression()
model.fit(X, y)
y_pred model.predict(X)
#Graph
plt.scatter (Х, у)
plt.plot(X, y_pred)
plt.xlabel("House Size (sq.ft)") plt.ylabel("House Price (Lakhs)")
plt.title("House Price Prediction using Linear Regression")
plt.show()

new_size np.array([[1600]])
predicted_price model.predict(new_size)
print("Predicted price for 1600 sq.ft house:", predicted_price[0], "Lakhs")
