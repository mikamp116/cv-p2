import os
import sys
import random
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import deteccion_haar as haardet


def coordenada_x(elem):
    return elem[0]


def load(directory, color=False, exclude=None):
    """Devuelve una lista con las imagenes en color o escala de grises contenidas en el directorio pasado como argumento
    descartando aquellas imagenes cuya primera letra este en la lista de excluidos"""
    if exclude is None:
        exclude = ['.']
    cur_dir = os.path.abspath(os.curdir)
    with os.scandir(cur_dir + '/' + directory) as it:
        files = [file.name for file in it if file.name[0] not in exclude and file.is_file()]
    it.close()
    files.sort()
    if color is True:
        return [cv.imread(directory + '/' + file) for file in files], files
    else:
        return [cv.imread(directory + '/' + file, 0) for file in files], files


def umbralizado(images, blur=False, tipo=0, ksize=5, c=2):
    """Devuelve una lista de imagenes umbralizadas mediante el tipo de umbralizado especificado en los parámetros"""
    imagenes_umbralizadas = []
    for i in range(len(images)):
        image = images[i]

        if blur is True:
            image = cv.GaussianBlur(image, (3, 7), 0)

        if tipo == 0:
            th = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, ksize, c)
        elif tipo == 1:
            _, th = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
        else:
            _, th = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        imagenes_umbralizadas.append(th)

    return imagenes_umbralizadas


def umbralizado_lista(images, blur=False, tipo=0, ksize=5, c=2):
    """Devuelve una lista de imagenes umbralizadas mediante el tipo de umbralizado especificado en los parámetros"""
    imagenes_umbralizadas = []
    for image_list in images:
        imagenes_umbralizadas.append(umbralizado(image_list, blur, tipo, ksize, c))

    return imagenes_umbralizadas


def negativo(images):
    images_negativo = []
    for image_list in images:
        aux = []
        for image in image_list:
            aux.append(cv.bitwise_not(image))
        images_negativo.append(aux)

    return images_negativo


# Metodo inutil
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
    """Devuelve una lista con la posion de la matricula"""
    clasificador_matriculas = haardet.HaarDetector('haar_opencv_4.1-4.2/matriculas.xml')
    return clasificador_matriculas.detect(input_images, 1.1, 5)


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


def get_contornos_caracteres_list(images):
    contornos = []
    for image_list in images:
        contornos.append(get_contornos_caracteres(image_list))

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


def localizar(directory):
    input_images, _ = load(directory)
    input_images_color, _ = load(directory, color=True)
    # umbral = umbralizado(input_images)
    matriculas = get_contorno_matricula_haar(input_images)

    roi_matricula = []
    for (img, mat) in zip(input_images, matriculas):
        aux = []
        for (x, y, w, h) in mat:
            aux.append(img[y:y + h, x:x + w])
        roi_matricula.append(aux)

    roi_matricula_color = [img[mat[0, 1]:mat[0, 1] + mat[0, 3], mat[0, 0]:mat[0, 0] + mat[0, 2]] for (img, mat) in
                           zip(input_images_color, matriculas)]

    matriculas_umbral = umbralizado_lista(roi_matricula, blur=False, tipo=0, ksize=7, c=5)

    matriculas_umbral_inv = negativo(matriculas_umbral)

    caracteres = get_contornos_caracteres_list(matriculas_umbral_inv)
    to_return = []
    for (coche, img) in zip(caracteres, roi_matricula_color):
        aux_coche = []
        for matricula in coche:
            aux_matricula = []
            reg = []
            for (x, y, w, h) in matricula:
                caracter = img[y:y + h, x:x + w, :]
                a = np.std(caracter[:, :, 0])
                reg.append((a, (x, y, w, h)))

            if (len(reg) < 8):
                for (_, (x, y, w, h)) in reg:
                    aux_matricula.append((x, y, w, h))
                    img = cv.rectangle(img, (x, y), (x + w, y + h),
                                       (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)  # rojo
            else:
                reg.sort(key=take_first, reverse=True)
                for (_, (x, y, w, h)) in reg[:7]:
                    aux_matricula.append((x, y, w, h))
                    img = cv.rectangle(img, (x, y), (x + w, y + h),
                                       (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)  # rojo

            aux_matricula.sort(key=coordenada_x)
            aux_coche.append(aux_matricula)

        to_return.append(aux_coche)

    rels = []
    for (image_list, coche) in zip(roi_matricula, to_return):
        aux = []
        for (image, mat) in zip(image_list, coche):
            aux.append([image[y:y + h, x:x + w] for (x, y, w, h) in mat])
        rels.append(aux)

    rels_umbral = []
    for li in range(len(rels)):
        rels_umbral.append(umbralizado_lista(rels[li], False, 2))

    rels_umbral_inv = []
    for li in rels_umbral:
        rels_umbral_inv.append(negativo(li))

    return rels_umbral_inv, matriculas, to_return


if __name__ == "__main__":
    localizar('testing_ocr')
