import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import deteccion_orb
import deteccion_haar as haardet

def loadone():
    return cv2.imread('testing_full_system/test3.jpg', 0),\
           cv2.imread('testing_full_system/test3.jpg')

def load():
    training_list = sorted(os.listdir('testing_full_system'))
    return [cv2.imread('testing_full_system/' + img, 0) for img in training_list]

def load_color():
    training_list = sorted(os.listdir('testing_full_system'))
    return [cv2.imread('testing_full_system/' + img) for img in training_list]

def close_to_centre(car_x, car_y, rect_x, rect_y):
    return -20 < (rect_y - car_y) < 150 and -150 < (rect_x - car_x) < 150

def possible_plate_position(car_x, car_y, rect_x, rect_y):
    return -150 < (car_x - rect_x) < 100 and -50 < (rect_y - car_y) < 200

def possible_plate(car_pos, rect_dim):
    return possible_plate_position(car_pos[0], car_pos[1], rect_dim[0], rect_dim[1]+rect_dim[3]/2) \
           and minimum_plate_size(rect_dim[2], rect_dim[3])

def minimum_plate_size(width, height):
    return width > 40 and height > 9 and 4.2 < width/height < 5.2

def umbralizado(images, blur=False, tipo=0, ksize=11, c=3):
    """Devuelve una lista de imagenes umbralizadas mediante el tipo de umbralizado especificado en los parámetros"""
    imagenes_umbralizadas = []
    for i in range(len(images)):
        image = images[i]

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

        imagenes_umbralizadas.append(th)

    return imagenes_umbralizadas

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

def find_plate_contours(binary_img, color_img, car_position):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    plate_rect = []
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
                # color_img = cv2.rectangle(color_img, (x_p, y_p), (x_p + w_p, y_p + h_p), (0, 255, 255), 2)
                color_img = cv2.drawContours(color_img, [box], 0, (0, 0, 255), 2)

    return plate_rect

def get_possible_plates(img, color, car_position, erode=False, esize=3):
    threshold = umbralizado([img],blur=True, tipo=3, ksize=11, c=2)
    thresh_img = threshold[0]
    # plt.imshow(thresh_img, cmap='gray')
    # plt.title('binary')
    # plt.show()
    plates = []
    plates = find_plate_contours(thresh_img, color, car_position)

    color = cv2.circle(color, car_position, 5, (255, 0, 0), thickness=2, lineType=8, shift=0)
    # plt.imshow(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    # plt.title('possible plates')
    # plt.show()

    if len(plates) == 0 or erode:
        kernel = np.ones((esize, esize), np.uint8)
        eroded_img = cv2.erode(thresh_img, kernel, iterations=1)
        # plt.imshow(eroded_img, cmap='gray')
        # plt.title('binary')
        # plt.show()
        plates = find_plate_contours(eroded_img, color, car_position)

    return plates

def get_external_rectangles(aux):
    aux.sort(key=lambda t: t[0])
    aux2 = []
    if len(aux) > 1:
        aux2.append(aux[0])
    for i in range(len(aux) - 1):
        if aux[i][0] <= aux[i + 1][0] < aux[i][0] + aux[i][2] and aux[i + 1][0] - aux[i][0] + aux[i + 1][2] < \
                aux[i + 1][0]:
            pass
        else:
            aux2.append(aux[i + 1])
    aux3 = []
    if len(aux) > 1:
        aux3.append(aux2[len(aux2) - 1])
    for i in range(len(aux2) - 1):
        if aux2[i + 1][0] <= aux2[i][0] < aux2[i + 1][0] + aux2[i + 1][2] and aux2[i][0] - aux2[i + 1][0] + aux2[i][2] < \
                aux2[i][0]:
            pass
        else:
            aux3.append(aux[i])
    return aux3

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


def get_contorno_matricula_haar(input_images):
    """Devuelve una lista con la posion de la matricula"""
    clasificador_matriculas = haardet.HaarDetector('haar_opencv_4.1-4.2/matriculas.xml')
    return clasificador_matriculas.detect(input_images, 1.1, 5)

def find_numbers_in_plates(img, color, possible_plates, rotated_plate=False):
    possible_plates = np.array(possible_plates)
    data_x = []
    data_y = []
    total_chars = []
    plate_with_chars = []
    chars_found = []

    min_wh_ratio = 1.5 if rotated_plate else 1
    max_wh_ratio = 6 if rotated_plate else 5

    for plate in possible_plates:
        chars_in_plate = []
        plate_img = img[plate[1]:plate[1] + plate[3], plate[0]:plate[0] + plate[2]]
        thresh_img = umbralizado([plate_img], blur=True)
        # plt.imshow(thresh_img[0], cmap='gray')
        # plt.title('binary')
        # plt.show()
        contours, _ = cv2.findContours(thresh_img[0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 15:
                x, y, w, h = cv2.boundingRect(cnt)
                if min_wh_ratio < h/w < max_wh_ratio:
                    chars_in_plate.append((x,y,w,h))
                    # color = cv2.rectangle(color, (plate[0]+x, plate[1]+y), (plate[0]+x + w, plate[1]+y + h), (255, 255, 0))
        chars_in_plate = get_external_rectangles2(chars_in_plate)

        chars_found.append(len(chars_in_plate))
        total_chars.append(chars_in_plate)

    if len(chars_found) > 0:

        most_chars_plate_index = np.argmax(chars_found)
        plate_with_chars = possible_plates[most_chars_plate_index]
        chars_in_plate = total_chars[most_chars_plate_index]

        for char in chars_in_plate:
            # data_x.append(number[0]+number[2]/2)
            # data_y.append(number[1]+number[3]/2)
            x = plate_with_chars[0] + char[0]
            y = plate_with_chars[1] + char[1]
            w = char[2]
            h = char[3]
            color = cv2.rectangle(color, (x, y), (x + w, y + h), (255, 255, 0))


            # plt.imshow(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            # plt.title('possible numbers')
            # plt.show()

            # if len(chars_in_plate) > 8:
            #     numbers_in_plate = ransac_numbers_in_plate(chars_in_plate, data_x, data_y, color)
            #
            # for rect in chars_in_plate:
            #     x, y, w, h = rect
            #     color = cv2.rectangle(color, (x, y), (x + w, y + h), (0, 255, 50))
    else:
        chars_in_plate = []

    plt.imshow(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    plt.title('numbers in plate')
    plt.show()

    return chars_in_plate

    # else:
    #     # plt.imshow(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    #     # plt.title('no plates found')
    #     # plt.show()
    #
    #     return []

def ransac_numbers_in_plate(numbers, data_X, data_Y, color_img):

    data_x = np.array(data_X)
    data_y = np.array(data_Y)

    data_x = data_x.reshape(-1, 1)
    data_y = data_y.reshape(-1, 1)

    if len(numbers) > 2:

        ransac = linear_model.RANSACRegressor()
        ransac.fit(data_x, data_y)

        line_x = np.arange(data_x.min(), data_x.max())[:, np.newaxis]

        line_y_ransac = ransac.predict(line_x)

        print(ransac.score(data_x, data_y))

        plt.plot(line_x, line_y_ransac, color='cornflowerblue', linewidth=2)

        ransac_min = line_y_ransac.min()
        ransac_max = line_y_ransac.max()

        found_characters = []

        for rect in numbers:
            x, y, w, h = rect
            if ransac_min - 2 < y + h / 2 < ransac_max + 2:
                found_characters.append(rect)
                color_img = cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 50), 2)

        return found_characters

    else:
        return []


# 1 si el haar funciona, usar eso
# 2 si no, buscar matriculas con umbralizado adaptativo
# si dentro no hay entre 5 y 9 caracteres, volver a buscar matriculas erosionando la imagen


def test_images():
    train_images = deteccion_orb.load()
    orb = cv2.ORB_create(nfeatures=300, scaleFactor=1.3, nlevels=4)
    match_table, flann = deteccion_orb.train(train_images, orb)

    images = load()
    color_images = load_color()
    matriculas_haar = get_contorno_matricula_haar(images)
    caracteres = []

    for i in range(len(images)):
        # primero intentamos buscar la matricula con haar
        if len(matriculas_haar[i]) > 0:
            numbers = find_numbers_in_plates(images[i], color_images[i], matriculas_haar[i])
        else:
            detected_points = deteccion_orb.detect([images[i]], orb, match_table, flann, 4, 2, 1)
            plates = get_possible_plates(images[i], color_images[i], detected_points[0])
            numbers = find_numbers_in_plates(images[i], color_images[i], plates, rotated_plate=True)
            for j in range (2,5):
                if len(numbers) < 4:
                    plates = get_possible_plates(images[i], color_images[i], detected_points[0], erode=True, esize=j)
                    numbers = find_numbers_in_plates(images[i], color_images[i], plates, rotated_plate=True)
        caracteres.append(numbers)
        color_img = color_images[i]


    return caracteres

def test_image():
    train_images = deteccion_orb.load()
    orb = cv2.ORB_create(nfeatures=300, scaleFactor=1.3, nlevels=4)
    match_table, flann = deteccion_orb.train(train_images, orb)
    image, color = loadone()
    detected_points = deteccion_orb.detect([image], orb, match_table, flann, 4, 2, 1)
    plates = get_possible_plates(image, color, detected_points[0])
    find_numbers_in_plates(image, color, plates)

if __name__ == '__main__':

    numeritos = test_images()
    print(numeritos)