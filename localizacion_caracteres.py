import os
import numpy as np
import cv2 as cv
import deteccion_haar as haardet
import deteccion_orb
import localizacion_matricula


def coordenada_x(elem):
    return elem[0]


def load(directory, color=False, exclude=None):
    """Devuelve una lista con las imagenes en color o escala de grises contenidas en el directorio pasado como argumento
    descartando aquellas imagenes cuya primera letra este en la lista de excluidos"""
    if exclude is None:
        exclude = ['.']
    cur_dir = os.path.abspath(os.curdir)
    if directory.startswith("/"):
        path = directory
    else:
        path = cur_dir + '/' + directory
    with os.scandir(path) as it:
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


def get_contorno_matricula_haar(input_images):
    """Devuelve una lista con la posion de la matricula"""
    clasificador_matriculas = haardet.HaarDetector('haar_opencv_4.1-4.2/matriculas.xml')
    return clasificador_matriculas.detect(input_images, 1.1, 5)


def localizar(directory):
    # Imagenes sobre las que detectar en escala de grises. Por cada imagen hay una lista con un elemento por cada
    # matricula detectada
    input_images, _ = load(directory)
    # Lo mismo en color
    input_images_color, _ = load(directory, color=True)
    # Lista de coordenadas de las matriculas. Cada array tiene tantas filas como matriculas hay en la imagen
    matriculas = get_contorno_matricula_haar(input_images)

    train_images = deteccion_orb.load()
    orb = cv.ORB_create(nfeatures=300, scaleFactor=1.3, nlevels=4)
    match_table, flann = deteccion_orb.train(train_images, orb)

    matriculas_total = []
    caracteres = []

    for i in range(len(matriculas)):
        if len(matriculas[i]) > 0:
            numbers, _ = localizacion_matricula.find_numbers_in_plates(input_images[i], matriculas[i])
            rect_plate = matriculas[i]
            rect_plate = rect_plate[0]
            caracteres.append([numbers])
            matriculas_total.append([rect_plate])
        else:
            detected_points = deteccion_orb.detect([input_images[i]], orb, match_table, flann, 4, 2, 1)
            centre = detected_points[0]
            rect_plates, box_plates = localizacion_matricula.get_possible_plates(input_images[i], centre)
            numbers = localizacion_matricula.find_numbers_in_plates(input_images[i], rect_plates, rotated_plate=True)
            for j in range(2, 5):
                if len(numbers) < 4:
                    rect_plates, box_plates = localizacion_matricula.get_possible_plates(input_images[i], centre,
                                                                                         erode=True, esize=j)
                    numbers, plate_index = localizacion_matricula.find_numbers_in_plates(input_images[i], rect_plates,
                                                                                         rotated_plate=True)
            if len(rect_plates) > 0:
                rect_plate = rect_plates[plate_index]
            caracteres.append([numbers])
            matriculas_total.append(np.array([rect_plate]))

    # Lista de regiones de las imagenes conteniendo las matriculas
    roi_matricula = []
    for (img, mat) in zip(input_images, matriculas_total):
        aux = []
        for (x, y, w, h) in mat:
            aux.append(img[y:y + h, x:x + w])
        roi_matricula.append(aux)

    # De todos los caracteres queremos obtener 7, ya que tambien detecta la E y las sobras de izquierda y derecha
    caracteristicas_seleccionadas = []
    for coche in caracteres:
        aux_coche = []
        for matricula in coche:
            aux_matricula = []
            medias = []
            for (x, y, w, h) in matricula:
                medias.append((h, (x, y, w, h)))

            if (len(medias) < 8):
                for (_, (x, y, w, h)) in medias:
                    aux_matricula.append((x, y, w, h))
            else:
                mean_y = 0
                for (y, _) in medias:
                    mean_y += y
                mean_y = mean_y / len(medias)

                medias.sort(key=lambda r: np.abs(r[0] - mean_y))
                for (_, (x, y, w, h)) in medias[:7]:
                    aux_matricula.append((x, y, w, h))

            aux_matricula.sort(key=coordenada_x)
            aux_coche.append(aux_matricula)

        caracteristicas_seleccionadas.append(aux_coche)

    # Obtiene las secciones de imagen con los caracteres
    roi_caracteres = []
    for (image_list, coche) in zip(roi_matricula, caracteristicas_seleccionadas):
        aux = []
        for (image, mat) in zip(image_list, coche):
            aux.append([image[y:y + h, x:x + w] for (x, y, w, h) in mat])
        roi_caracteres.append(aux)

    # Las vuelve a unmbralizar e invertir
    roi_caracteres_umbral = []
    for li in range(len(roi_caracteres)):
        roi_caracteres_umbral.append(umbralizado_lista(roi_caracteres[li], False, 2))

    roi_caracteres_umbral_inv = []
    for li in roi_caracteres_umbral:
        roi_caracteres_umbral_inv.append(negativo(li))

    return roi_caracteres_umbral_inv, matriculas_total, caracteristicas_seleccionadas


if __name__ == "__main__":
    localizar('testing_ocr')
