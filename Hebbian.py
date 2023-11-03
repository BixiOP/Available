import numpy as np

features = np.array([
    [-1, -1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, 1]
])

lables = np.array([-1, -1, -1, 1])

epochs = features.shape[0]

weights = [0 for _ in range(features.shape[1])]

for i in range(epochs):
  weights += features[i]*lables[i]

def predict(x):
  return np.dot(weights.T, x)

for i in range(features.shape[0]):
  print("[x1 x2 b] =", features[i], ", [w1 w2 w3] =", weights, ", prediction =", predict(features[i]))