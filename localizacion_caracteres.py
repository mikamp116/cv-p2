import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import sys
import deteccion_haar as haardet
import random


def coordenada_x(elem):
    return elem[0]


# def load(directory, color=False, exclude=None):
#     """Recibe el nombre de un directorio y devuelve una lista con las imagenes contenidas en el"""
#     # https://stackoverflow.com/questions/51520/how-to-get-an-absolute-file-path-in-python#51523
#     if exclude is None:
#         exclude = ['.']
#     cur_dir = os.path.abspath(os.curdir)
#     files = os.listdir(cur_dir + '/' + directory)
#     files.sort()
#     if color is True:
#         return [cv.imread(directory + '/' + file) for file in files if file[0] not in exclude]
#     else:
#         return [cv.imread(directory + '/' + file, 0) for file in files if file[0] not in exclude]


def load(directory, color=False, exclude=None):
    """Recibe el nombre de un directorio y devuelve una lista con las imagenes contenidas en el"""
    # https://stackoverflow.com/questions/51520/how-to-get-an-absolute-file-path-in-python#51523
    if exclude is None:
        exclude = ['.']
    cur_dir = os.path.abspath(os.curdir)
    with os.scandir(cur_dir + '/' + directory) as it:
        files = [file.name for file in it if file.name[0] not in exclude and file.is_file()]
    it.close()
    files.sort()
    if color is True:
        return [cv.imread(directory + '/' + file) for file in files]
    else:
        return [cv.imread(directory + '/' + file, 0) for file in files]


def umbralizado(images, blur=False, tipo=0, ksize=5, c=2):
    imagenes_umbralizadas = []
    for i in range(len(images)):
        image = images[i]

        if blur is True:
            image = cv.GaussianBlur(image, (3, 7), 0)

        if (tipo==0):
            th = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, ksize, c)
        elif (tipo==1):
            _, th = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
        else:
            _, th = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        #ret4, th4 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # plt.imshow(th, "gray")
        # plt.show()
        # plt.imshow(th2, "gray")
        # plt.show()
        # plt.imshow(th3, "gray")
        # plt.show()
        # plt.imshow(th4, "gray")
        # plt.show()

        imagenes_umbralizadas.append(th)

    return imagenes_umbralizadas


def pintar_matriculas(input_images):
    clasificador_matriculas = haardet.HaarDetector('haar_opencv_4.1-4.2/matriculas.xml')
    matriculas = clasificador_matriculas.detect(input_images)
    for i in range(len(matriculas)):
        image = input_images[i]
        for (x, y, w, h) in matriculas[i]:
            image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        plt.imshow(image)
        plt.show()


def get_contorno_matricula_haar(input_images):
    clasificador_matriculas = haardet.HaarDetector('haar_opencv_4.1-4.2/matriculas.xml')
    return clasificador_matriculas.detect2(input_images, 1.1, 5)


def take_first(elem):
    return elem[0]


def get_contornos_caracteres(images):
    """Devuelve una lista con los caracteres encontrados. Cada posicion de la lista es una lista con cada caracter"""
    contornos = []
    for image in images:
        aux = []
        contours, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if h > 1.1 * w and h > image.shape[0] * 0.4:
                if x > image.shape[1] * 0.6:
                    if h < 3 * w:
                        aux.append((x, y, w, h))
                else:
                    aux.append((x, y, w, h))
        contornos.append(aux)

    return contornos


def get_contornos_matricula(images):
    contornos = []
    for image in images:
        aux = []
        contours, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if 1.5 * h < w < 7 * h:
                aux.append((x, y, w, h))
        contornos.append(aux)

    return contornos


def localizar():
    input_images = load('testing_ocr')
    color_images = load_color('testing_ocr')
    umbral = umbralizado(input_images)
    matriculas = get_contorno_matricula_haar(input_images)

    # for i in range(len(color_images)):
    #     image = color_images[i]
    #     contornos_matricula = matriculas[i]
    #     for (x, y, w, h) in contornos_matricula:
    #         image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     plt.imshow(image)
    #     plt.show()

    roi_matricula = [img[mat[0, 1]:mat[0, 1] + mat[0, 3], mat[0, 0]:mat[0, 0] + mat[0, 2]] for (img, mat) in
                     zip(input_images, matriculas)]
    roi_matricula_color = [img[mat[0, 1]:mat[0, 1] + mat[0, 3], mat[0, 0]:mat[0, 0] + mat[0, 2]] for (img, mat) in
                           zip(color_images, matriculas)]
    """for image in roi_matricula:
        plt.imshow(image)
        plt.show()"""

    matriculas_umbral = umbralizado(roi_matricula, 7, 5)
    """for m in matriculas_umbral:
        plt.imshow(m, "gray")
        plt.show()"""

    caracteres = get_contornos_caracteres(matriculas_umbral)
    to_return = []
    for (matricula, img) in zip(caracteres, roi_matricula_color):
        aux = []
        reg = []
        for (x, y, w, h) in matricula:
            caracter = img[y:y + h, x:x + w, :]
            a = np.std(caracter[:, :, 0])
            reg.append((a, (x, y, w, h)))

        if (len(reg) < 8):
            for (_, (x, y, w, h)) in reg:
                aux.append((x, y, w, h))
                img = cv.rectangle(img, (x, y), (x + w, y + h), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)  # rojo
        else:
            reg.sort(key=take_first, reverse=True)
            for (_, (x, y, w, h)) in reg[:7]:
                aux.append((x, y, w, h))
                img = cv.rectangle(img, (x, y), (x + w, y + h), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)  # rojo

        plt.imshow(img)
        plt.show()
        aux.sort(key=coordenada_x())
        to_return.append(([aux[:3]], [aux[3:]]))

    return to_return


if __name__ == "__main__":
    localizar()
