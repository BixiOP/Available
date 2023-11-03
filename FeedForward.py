import numpy as np

def sigmoid(x):
  return 1/(1+np.exp(-x))

features = np.array([1, 0])

lables = np.array([1])

weights1 = np.array([[0.2, -0.3], [0.4, 0.1]])
weights2 = np.array([-0.5, 0.2])

bias1 = np.array([0.2, 0.1])
bias2 = np.array([0.1])

hidden_output = np.array([0.0, 0.0])

for i in range(weights1.shape[1]):
  for j in range(weights1.shape[0]):
    hidden_output[i] += features[j]*weights1[j][i]
  hidden_output[i] += bias1[i]
  hidden_output[i] = sigmoid(hidden_output[i])

output = np.dot(weights2.T, hidden_output) + bias2
print(output)

error = sum(lables - output)
print("error:", error)

mean_squared_error = 0.5*(error**2)
print("mean squared error: ", mean_squared_error)