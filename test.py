import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
model = joblib.load('iris_model.pkl')
testdf = pd.read_csv('Iris - Iris.csv')
testdf = testdf.drop('Id', axis=1)
x=testdf.drop('Species', axis=1)
pred=model.predict(x)
y=testdf['Species']
print("Predictions:", pred)
print("Actual:", y)
acc=accuracy_score(pred,y)
print(acc) 
assert len(pred) == len(y)
assert acc >= 0.80