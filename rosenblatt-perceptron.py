import argparse

from perceptron import Perceptron
from image_loader import ShapeMaker
from train import train

import cv2
import numpy as np


def toFloat(image):

        image = image.astype('float32')
        norm = np.linalg.norm(image)
        image /= norm

        return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--bias", help="The Bias for the Perceptron")
    parser.add_argument("-t", "--train", help="1 if you want to train the model, otherwise 0")

    parser.add_argument("-i", "--image", help="Path to the image file to classfy, only applicable if --train is 0. Only works if the image is a 128 x 128 image")
    parser.add_argument("-n", "--num_epochs", help="Number of images to train on, only applicable if --train is 1")

    args = parser.parse_args()

    net = Perceptron(int(args.bias))
    sm = ShapeMaker()

    if args.train == "1":
        train(net, sm, int(args.num_epochs))

    if args.train == "0":

        net.load()
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

        img = toFloat(img)

        result = net.forward(img)

        if result == 1:
            print("The input is a Circle")

        else:
            print("The input is a Rectangle")