import numpy
import cv2 as cv
from matplotlib import pyplot as plt
import os
import sys


def load(directory):
    """Recibe el nombre de un directorio y devuelve una lista con las imagenes contenidas en el"""
    # https://stackoverflow.com/questions/51520/how-to-get-an-absolute-file-path-in-python#51523
    cur_dir = os.path.abspath(os.curdir)
    files = os.listdir(cur_dir + '/' + directory)
    return [cv.imread(directory + '/' + file, 0) for file in files if not file.startswith('.')]


def main():
    print(sys.version)
    print(cv.__version__)
    input_images = load('testing_ocr')
    for image in input_images:
        th = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        ret2, th2 = cv.threshold(image,127,255,cv.THRESH_BINARY)
        # Otsu's thresholding
        ret3, th3 = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # Otsu's thresholding after Gaussian filtering
        blur = cv.GaussianBlur(image, (5, 5), 0)
        ret4, th4 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        plt.imshow(th, "gray")
        plt.show()
        plt.imshow(th2, "gray")
        plt.show()
        plt.imshow(th3, "gray")
        plt.show()
        plt.imshow(th4, "gray")
        plt.show()
    # for i in range(28):
    #     plt.imshow(input[i], "gray")
    #     plt.show()


if __name__ == "__main__":
    main()