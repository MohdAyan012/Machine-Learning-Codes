import pandas as pd
from sklearn.linear_model import LogisticRegression
#dataset
data = {
    'Age': [22, 25, 47, 35, 46],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Chennai', 'Mumbai'],
    'Bought': [0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=['City'], drop_first=True)
print("Dataset after dummy variable conversion:")
print(df)
X = df.drop('Bought', axis=1)
y = df['Bought']
model = LogisticRegression()
model.fit(X, y)
new_data = pd.DataFrame({
    'Age': [40],
    'City_Delhi': [0],
    'City_Mumbai': [1]})
prediction = model.predict(new_data)
print("Prediction (1 = Buy, 0 = Not Buy):", prediction[0])
