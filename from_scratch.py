from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np

data = load_diabetes()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.columns)

X = df.drop(columns='target')
y = df['target']

indices = np.random.permutation(len(X))

test_size = int(len(X) * 0.2)

train_indices = indices[test_size:]
test_indices = indices[:test_size]


x_train = X.iloc[train_indices]
x_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

mean = np.mean(x_train)
std = np.std(x_train)

x_train_scaled = (x_train - mean) / std
x_test_scaled = (x_test - mean) / std

x_train_scaled = x_train_scaled.to_numpy()
x_test_scaled = x_test_scaled.to_numpy()


def euclidean_distance(x_test, x_train):
   distance = 0
   for i in range(len(x_train)):
      distance += (x_train[i] - x_test[i])**2
   return np.sqrt(distance)

def calculate_distance(x_test, x_train):
   distances = []
   for point in x_train:
      distances.append(euclidean_distance(x_test, point))
   return distances

def labeling_distance(distances, y_train):
   labels = []
   for d, l in zip(distances, y_train):
      labels.append((d,l))
   return labels

def sorted_distance(labels):
   return sorted(labels)

def nearest_distances(k, sorts):
   return sorts[:k]

def average(nearest):
   values = [val for distance, val in nearest]
   return np.mean(values)

def knn_predict(k, x_train, x_test, y_train):
   distances = calculate_distance(x_test, x_train)
   labeled = labeling_distance(distances, y_train)
   sorted_ = sorted_distance(labeled)
   nearest = nearest_distances(k, sorted_)

   return average(nearest)
best_k = 5
def predict(k, x_train_scaled, x_test_scaled, y_train):
   predictions = []
   for point in x_test_scaled:
      predictions.append(knn_predict(k, x_train_scaled, point, y_train))
   return predictions
predictions = predict(best_k, x_train_scaled, x_test_scaled, y_train)

def MAE(predictions, y_test):
   error = 0
   for p, t in zip(predictions, y_test):
      error += abs(p - t)
   return error / len(y_test)
test_MAE = MAE(predictions, y_test)
print('Final test MAE:', test_MAE)


