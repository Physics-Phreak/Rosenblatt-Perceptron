import numpy as np
import cv2

class Perceptron():

    def __init__(self, bias):

        self.weights = np.zeros((128, 128))
        self.bias = bias

    def forward(self, image):

        pass_1 = image * self.weights
        pass_2 = np.sum(pass_1)

        return 1 if pass_2 > self.bias else 0

    def train(self, image, er):

        result = self.forward(image)

        if result == 1 and er == 0:
            self.weights = np.subtract(self.weights, image)

        if result == 0 and er == 1:
            self.weights = np.add(self.weights, image)

        return result

    def save(self):

        np.save("weights.npy", self.weights)

    def load(self):
        self.weights = np.load("weights.npy")

    def showWeight(self):
        cv2.imshow("Weights", self.weights)

        cv2.waitKey(0)
        cv2.destroyAllWindows()