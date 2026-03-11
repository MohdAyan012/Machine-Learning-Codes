
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
marks = np.array([30, 35, 50, 55, 65, 70, 80, 90])

model = LinearRegression()
model.fit(hours, marks)

predicted = model.predict([[9]])
print("Predicted Marks for 9 hours of study:", predicted[0])

plt.scatter(hours, marks)
plt.plot(hours, model.predict(hours))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.show()
