import numpy as np
import cv2

import random

class ShapeMaker():

    def getShape(self, shape):

        img = np.zeros((128, 128))

        if shape == 0:

            top_left_x = random.randint(1, 128)
            top_left_y = random.randint(1, 128)


            length_x = random.randint(4, 64)
            length_y = random.randint(4, 64)

            rect = cv2.rectangle(img, (top_left_x, top_left_y), (top_left_x + length_x, top_left_y + length_y), 255, cv2.FILLED)

            return self.toFloat(rect)

        if shape == 1:

            center_x = random.randint(1, 128)
            center_y = random.randint(1, 128)

            radius = random.randint(4, 64)

            circ = cv2.circle(img, (center_x, center_y), radius, 255, cv2.FILLED)

            return self.toFloat(circ)

    def toFloat(self, image):

        image = image.astype('float32')
        norm = np.linalg.norm(image)
        image /= norm

        return image