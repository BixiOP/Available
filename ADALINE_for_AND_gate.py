import numpy as np

class AndNN:
    def __init__(self):

      self.features = np.array(
      [
          [-1, -1],
          [-1, 1],
          [1, -1],
          [1, 1]
      ])

      self.lables = [-1, -1, -1, 1]

      self.w = np.array([0.5, 0.5])
      self.b = 0.1
      self.alpha = 0.2
      self.epochs = 10

      for i in range(self.epochs):
        print("epoch:", i+1)
        sum_squared_error = 0.0

        for j in range(self.features.shape[0]):
          actual = self.lables[j]
          x = self.features[j]
          unit = np.dot(self.w,x) + self.b
          error = actual - unit
          print("error:", error)
          sum_squared_error += error**2
          self.w += self.alpha*error*x
          self.b += self.alpha*error
        print("sum_squared_error:", sum_squared_error/4, "\n\n")

    def predict(self, x):
      return np.dot(self.w, x) + self.b

NN = AndNN()
x = [int(i) for i in input("Enter the input to be predicted: ").split()]
print("prediction:", NN.predict(x))