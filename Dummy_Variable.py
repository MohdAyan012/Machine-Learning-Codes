import pandas as pd
from sklearn.linear_model import LinearRegression
#dataset
data = {
    'Experience': [1, 2, 3, 4],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Chennai'],
    'Salary': [20000, 25000, 30000, 35000]}
df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=['City'], drop_first=True)
print("Dataset after Dummy Variable Conversion:")
print(df)

X = df.drop('Salary', axis=1)
y = df['Salary']
model = LinearRegression()
model.fit(X, y)
# Predict salary for new data
new_data = pd.DataFrame({
    'Experience': [5],
    'City_Delhi': [0],
    'City_Mumbai': [1]
})
prediction = model.predict(new_data)
print("\nPredicted Salary:", prediction[0])
