import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import deteccion_orb
import deteccion_haar as haardet
import caracteristicas_coche

def load_files():
    return sorted(os.listdir('testing_full_system'))

def load(files):
    return [cv2.imread('testing_full_system/' + img, 0) for img in files]

def load_color(files):
    return [cv2.imread('testing_full_system/' + img) for img in files]

"""Devuelve una lista con la posicion de las matriculas"""
def get_contorno_matricula_haar(input_images):
    clasificador_matriculas = haardet.HaarDetector('haar_opencv_4.1-4.2/matriculas.xml')
    return clasificador_matriculas.detect(input_images, 1.1, 5)

"""Devuelve una lista con la posicion de los coches"""
def get_centro_coche_haar(input_images):
    clasificador_matriculas = haardet.HaarDetector('haar_opencv_4.1-4.2/coches.xml')
    return clasificador_matriculas.detect(input_images, 1.1, 5)


def umbralizado(image, blur=False, tipo=0, ksize=11, c=3):
    """Devuelve una imagen umbralizada mediante el tipo de umbralizado especificado en los parámetros"""
    if blur is True:
        image = cv2.GaussianBlur(image, (5,5), 0)

    if tipo == 0:
        th = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ksize, c)
    elif tipo == 1:
        _, th = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    elif tipo == 2:
        _, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        th = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ksize, c)

    return th

def possible_plate_position(car_x, car_y, rect_x, rect_y):
    return -150 < (car_x - rect_x) < 100 and -50 < (rect_y - car_y) < 200

def possible_plate(car_pos, rect_dim):
    return possible_plate_position(car_pos[0], car_pos[1], rect_dim[0], rect_dim[1]+rect_dim[3]/2) \
           and minimum_plate_size(rect_dim[2], rect_dim[3])

def minimum_plate_size(width, height):
    return width > 40 and height > 9 and 4.2 < width/height < 5.2

def get_box_size(rect):
    top_left_corner = np.rint(rect[0])
    x, y = np.int0(top_left_corner)
    dimensions = np.rint(rect[1])
    # devolver altura y anchura dependiendo del angulo
    if rect[2] < -5:
        h, w = np.int0(dimensions)
    else:
        w, h = np.int0(dimensions)
    return x, y, w, h

"""Recibe una lista de rectangulos y devuelve otra que solo contiene rectangulos exteriores, es decir,
    que no son contenidos por otros"""
def get_external_rectangles2(aux):
    length = len(aux)
    external_rects = []
    for i in range(length):
        rect = aux[i]
        internal = False
        j = 0
        while not internal and j < length:
            if i != j:
                aux_rect = aux[j]
                internal = aux_rect[0] < rect[0] < aux_rect[0]+aux_rect[2] and aux_rect[1] < rect[1] < aux_rect[1]+aux_rect[3]
            j = j+1
        if not internal:
            external_rects.append(rect)
    return external_rects

"""A partir de una imagen umbralizada y la posicion del coche devuelve una tupla con las posibles matriculas
    plate_rect contiene los rectangulos paralelos a la imagen y plate_box los rectangulos exactos (pueden estar rotados) """
def find_plate_contours(binary_img, car_position):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    plate_rect = []
    plate_box = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 9:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x, y, w, h = get_box_size(rect)
            x_p, y_p, w_p, h_p = cv2.boundingRect(cnt)
            if possible_plate(car_position, (x, y, w, h)):
                plate_rect.append([x_p,y_p,w_p,h_p])
                plate_box.append(box)

    return plate_rect, plate_box

"""A partir de una imagen obtiene los posibles rectangulos de matriculas.
    si a la primera no encuentra nada o si la flag erode está activada,
    se erosiona la imagen umbralizada para encontrar la matricula"""
def get_possible_plates(img, car_position, erode=False, esize=3):
    threshold = umbralizado(img,blur=True, tipo=3, ksize=11, c=2)
    plates = []
    rect_plates, box_plates = find_plate_contours(threshold, car_position)


    if len(plates) == 0 or erode:
        kernel = np.ones((esize, esize), np.uint8)
        eroded_img = cv2.erode(threshold, kernel, iterations=1)
        rect_plates, box_plates = find_plate_contours(eroded_img, car_position)

    return rect_plates, box_plates

def find_numbers_in_plates(img, possible_plates, rotated_plate=False):
    possible_plates = np.array(possible_plates)
    total_chars = []
    chars_found = []
    min_wh_ratio = 1.5 if rotated_plate else 1
    max_wh_ratio = 6 if rotated_plate else 5

    for plate in possible_plates:
        chars_in_plate = []
        plate_img = img[plate[1]:plate[1] + plate[3], plate[0]:plate[0] + plate[2]]
        thresh_img = umbralizado(plate_img, blur=True)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 15:
                x, y, w, h = cv2.boundingRect(cnt)
                if min_wh_ratio < h/w < max_wh_ratio:
                    chars_in_plate.append((x,y,w,h))
        chars_in_plate = get_external_rectangles2(chars_in_plate)

        chars_found.append(len(chars_in_plate))
        total_chars.append(chars_in_plate)

    if len(chars_found) > 0:
        most_chars_plate_index = np.argmax(chars_found)
        chars_in_plate = total_chars[most_chars_plate_index]

    else:
        chars_in_plate = []
        most_chars_plate_index = 0


    return chars_in_plate, most_chars_plate_index

def buscar_matricula(image, orb, match_table, flann):
    detected_points = deteccion_orb.detect([image], orb, match_table, flann, 4, 2, 1)
    centre = detected_points[0]
    rect_plates, box_plates = get_possible_plates(image, centre)
    numbers = find_numbers_in_plates(image, rect_plates, rotated_plate=True)
    for j in range(2, 5):
        if len(numbers) < 4:
            rect_plates, box_plates = get_possible_plates(image, centre, erode=True, esize=j)
            numbers, plate_index = find_numbers_in_plates(image, rect_plates, rotated_plate=True)
    rect_plate = rect_plates[plate_index]
    return rect_plate

if __name__ == '__main__':
    train_images = deteccion_orb.load()
    orb = cv2.ORB_create(nfeatures=300, scaleFactor=1.3, nlevels=4)
    match_table, flann = deteccion_orb.train(train_images, orb)

    files = load_files()
    images = load(files)
    color_images = load_color(files)
    matriculas_haar = get_contorno_matricula_haar(images)
    caracteres = []
    coches = []

    for i in range(len(images)):
        # primero intentamos buscar la matricula con haar
        centre = 0
        if len(matriculas_haar[i]) > 0:
            numbers = find_numbers_in_plates(images[i], matriculas_haar[i])
            rect_plate = matriculas_haar[i]
            rect_plate = rect_plate[0]
            box_plate = []
            numbers = numbers[0][0]
        else:
            detected_points = deteccion_orb.detect([images[i]], orb, match_table, flann, 4, 2, 1)
            centre = detected_points[0]
            rect_plates, box_plates = get_possible_plates(images[i], centre)
            numbers = find_numbers_in_plates(images[i], rect_plates, rotated_plate=True)
            for j in range(2, 5):
                if len(numbers) < 4:
                    rect_plates, box_plates = get_possible_plates(images[i], centre, erode=True, esize=j)
                    numbers, plate_index = find_numbers_in_plates(images[i], rect_plates, rotated_plate=True)
            rect_plate = rect_plates[plate_index]
            rect_plate = rect_plate[0]
            box_plate = box_plates[plate_index]
        caracteres.append(numbers[0])
        coche = caracteristicas_coche.CarFeatures(files[i], images[i], color_images[i], centre, rect_plate, box_plate,
                                                  numbers)
        coche.mostrar()
        coches.append(coche)