from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np

data = load_diabetes()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.columns)

X = df.drop(columns='target')
y = df['target']

indices = 