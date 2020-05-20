import cv2
from matplotlib import pyplot as plt
import numpy as np

class CarFeatures:

    def __init__(self, title, grayscale_image, color_image, centre, frontal_plate, rotated_plate, plate_chars):
        self.title = title
        self.grayscale_image = grayscale_image
        self.color_image = color_image
        self.centre = centre
        self.frontal_plate = frontal_plate
        self.rotated_plate = rotated_plate
        self.plate_chars = plate_chars

    def mostrar(self):
        color = np.copy(self.color_image)
        if self.centre != 0:
            color = cv2.circle(color, self.centre, 5, (255, 0, 0), thickness=2, lineType=8, shift=0)
        color = cv2.rectangle(color, (self.frontal_plate[0], self.frontal_plate[1]), (self.frontal_plate[0] +
                                self.frontal_plate[2], self.frontal_plate[1] + self.frontal_plate[3]), (0, 255, 255), 2)
        if len(self.rotated_plate) > 0:
            color = cv2.drawContours(color, [self.rotated_plate], 0, (0, 0, 255), 2)
        for char in self.plate_chars:
            color = cv2.rectangle(color, (char[0], char[1]), (char[0] + char[2], char[1] + char[3]), (255, 255, 0))
        plt.imshow(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        plt.title(self.title)
        plt.show()