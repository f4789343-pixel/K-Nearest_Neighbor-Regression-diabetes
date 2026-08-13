from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

data = load_diabetes()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop(columns='target')
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


best_k = 5

model = KNeighborsRegressor(n_neighbors=best_k)
model.fit(x_train_scaled, y_train)

predictions = model.predict(x_test_scaled)

error = mean_absolute_error(predictions, y_test)

print('final test MAE:', error)

