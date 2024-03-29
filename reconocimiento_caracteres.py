"""
Apartado 2.\n
Modulo para el reconocimiento de caracteres en matriculas de coches
"""

import os
import numpy as np
import cv2 as cv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import localizacion_caracteres as localizacion


def get_contornos_caracteres(images):
    """Devuelve una lista con los caracteres encontrados. Cada posicion de la lista es una lista con cada caracter"""
    contornos = []
    for image in images:
        aux = []
        contours, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            aux.append((x, y, w, h))
        contornos.append(aux)

    return contornos


def area(rectangulo):
    """Devuelve el area de un rectangulo"""
    return rectangulo[2] * rectangulo[3]


class ReconocimientoCaracteres:
    """Reconocedor de caracteres en matriculas mediante LDA y GNB"""

    def __init__(self, included_characters, directory):
        """Crea un LDA y un GNB para el reconocimiento de caracteres"""
        self.included_characters = included_characters
        self.images_directory = directory
        self.lda = LinearDiscriminantAnalysis()
        self.gnb = GaussianNB()

    def __load(self, color=False, invert=True):
        """Devuelve una lista con las imagenes contenidas en el directorio de imagenes"""
        cur_dir = os.path.abspath(os.curdir)
        with os.scandir(cur_dir + '/' + self.images_directory) as it:
            files = [file.name for file in it if file.name[0] in self.included_characters and file.is_file()]
        it.close()
        files.sort()
        if color is True:
            return [cv.imread(self.images_directory + '/' + file) for file in files]
        else:
            if invert is True:
                return [cv.bitwise_not(cv.imread(self.images_directory + '/' + file, 0)) for file in files]
            else:
                return [cv.imread(self.images_directory + '/' + file, 0) for file in files]

    def entrenar(self):
        """Entrena un clasificador Naive Bayes para la deteccion de caracteres"""
        trains = self.__load()
        train_umbralizado = localizacion.umbralizado(trains, False, 2)
        traning_ocr_caracteres = get_contornos_caracteres(train_umbralizado)

        contador = np.zeros((len(self.included_characters)), dtype=np.uint16)

        roi_training_ocr = []
        for i in range(len(trains)):  # para cada imagen
            image = trains[i]
            caracteres = traning_ocr_caracteres[i]
            caracteres.sort(key=area, reverse=True)
            if len(caracteres) > 0:
                (x, y, w, h) = caracteres[0]
                roi_training_ocr.append(image[y:y + h, x:x + w])
                contador[i // 250] += 1

        E = [i for i in range(contador.shape[0]) for _ in range(contador[i])]

        traning_ocr_resized = [cv.resize(image, (10, 10), 0, 0, cv.INTER_LINEAR) for image in roi_training_ocr]

        C = np.array([char.reshape(1, 100).astype(np.float64) for char in traning_ocr_resized])[:, 0, :]

        self.lda.fit(C, E)
        CR = self.lda.transform(C)
        self.gnb.fit(CR, E)

        return C, E

    def reconocer(self, D):
        """Reconoce los caracteres de la matriz D"""
        DR = self.lda.transform(D)
        return self.gnb.predict(DR)
