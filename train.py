from perceptron import Perceptron
from image_loader import ShapeMaker

from tqdm import tqdm

import random

def train(net, sm, num_epochs):

    rect_total = 0
    rect_correct = 0

    circ_total = 0
    circ_correct = 0

    for _ in tqdm(range(num_epochs)):

        shape = random.randint(0, 1)

        img = sm.getShape(shape)

        result = net.train(img, shape)

        if shape == 1:
            circ_total += 1

        if shape == 0:
            rect_total += 1

        if result == shape:

            rect_correct += 1 if shape == 0 else 0
            circ_correct += 1 if shape == 1 else 0

    net.showWeight()

    rect = sm.getShape(0)
    circ = sm.getShape(1)

    res_rect = net.forward(rect)
    res_circ = net.forward(circ)

    print(res_rect, res_circ)

    print(f"Correct % for rectangle: {(rect_correct/rect_total) * 100}")
    print(f"Correct % for circle: {(circ_correct/circ_total) * 100}")

    net.save()

if __name__ == "__main__":

    sm = ShapeMaker()
    net = Perceptron(7)

    train(net, sm, 10_000_000)