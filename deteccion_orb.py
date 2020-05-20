import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
import math

assert (sys.version.startswith('3.7')), "No se esta usando la version 3.7 de Python. Version en uso: " + sys.version
assert (cv2.__version__.startswith('4.2')), "No se esta usando la version 4.2 de OpenCV. Version en uso: " + cv2.__version__


def ordenar(lst):
    """Recibe una lista y la devuelve ordenada"""
    lst.sort(key=len)
    ret = lst[0:10]
    ret.sort()
    aux = lst[10:]
    aux.sort()
    return ret + aux


def load(directory='train'):
    """Recibe el nombre de un directorio y devuelve una lista con las imagenes contenidas en el"""
    # https://stackoverflow.com/questions/51520/how-to-get-an-absolute-file-path-in-python#51523
    cur_dir = os.path.abspath(os.curdir)
    files = ordenar(os.listdir(cur_dir + '/' + directory))
    return [cv2.imread(directory + '/' + file, 0) for file in files]


def load_color(directory):
    """Recibe el nombre de un directorio y devuelve una lista con las imagenes contenidas en el a color"""
    cur_dir = os.path.abspath(os.curdir)
    files = ordenar(os.listdir(cur_dir + '/' + directory))
    return [cv2.imread(directory + '/' + file) for file in files]


def soft_load():
    """Devuelve una lista con 6 imagenes preseleccionadas aleatoriamente"""
    return [cv2.imread('train/frontal_9.jpg', 0), cv2.imread('train/frontal_39.jpg', 0),
            cv2.imread('train/frontal_43.jpg', 0), cv2.imread('train/frontal_7.jpg', 0),
            cv2.imread('train/frontal_19.jpg', 0), cv2.imread('train/frontal_26.jpg', 0)]


def load2():
    """Devuelve una lista con las 48 imagenes de entrenamiento"""
    return [cv2.imread('train/frontal_' + str(i) + '.jpg', 0) for i in range(1, 49)]


def calculate_module(p, centre=(225, 110)):
    """Recibe dos puntos y devuelve el modulo del vector que los une"""
    return np.sqrt((centre[0] - p[0]) ** 2 + (centre[1] - p[1]) ** 2)


def calculate_angle_to_centre(p, centre=(225, 110)):
    """Recibe dos puntos y devuelve el angulo del vector que los une"""
    return (math.atan2((p[1] - centre[1]), (centre[0] - p[0])) * 180 / math.pi) % 360


class Match:
    def __init__(self, module, kp_angle, scale, des_angle):
        self.module = module
        self.kp_angle = kp_angle
        self.scale = scale
        self.des_angle = des_angle

    def get_module(self):
        """Devuelve el modulo del vector que une el punto de interes con el centro de la imagen"""
        return self.module

    def get_kp_angle(self):
        """Devuelve el angulo del vector que une el punto de interes con el centro de la imagen"""
        return self.kp_angle

    def get_scale(self):
        """Devuelve la escala del punto de interes"""
        return self.scale

    def get_des_angle(self):
        """Devuelve el angulo del punto de interes respecto de la imagen"""
        return self.des_angle


def train(images, detector):
    """Devuelve una tabla con los puntos de interes aprendidos y un arbol flann entrenado"""
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=3, multi_probe_level=1)
    search_params = dict(checks=-1)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    match_table = []
    for image in images:
        kps, des = detector.detectAndCompute(image, None)
        image_match = [Match(calculate_module(k.pt), calculate_angle_to_centre(k.pt), k.size, k.angle) for k in kps]
        match_table.append(image_match)
        flann.add([des])

    return match_table, flann


def detect(images, detector, match_table, flann, knn_matches, sigma, debug):
    """Devuelve una lista de tuplas con las coordenadas de los puntos detectados"""
    if debug == 1:
        test_kps_table = []
        test_des_table = []
        matrices_votacion = []
    detected_points = []

    for test_image in images:
        kps, des = detector.detectAndCompute(test_image, None)
        if debug == 1:
            test_des_table.append(des)
            test_kps_table.append(kps)

        results = flann.knnMatch(des, k=knn_matches)

        matriz_votacion = np.zeros((int(test_image.shape[0] / 10), int(test_image.shape[1] / 10)), dtype=np.float32)

        for r in results:
            for m in r:
                match = match_table[m.imgIdx][m.trainIdx]
                m_test = kps[m.queryIdx]
                mod = (m_test.size / match.get_scale()) * match.get_module()
                angle = match.get_kp_angle() + match.get_des_angle() - m_test.angle
                x = int((m_test.pt[0] + (mod * math.cos(angle))) / 10)
                y = int((m_test.pt[1] - (mod * math.sin(angle))) / 10)
                if 0 < x < matriz_votacion.shape[1] and 0 < y < matriz_votacion.shape[0]:
                    matriz_votacion[y, x] += 1

        ksize = 6 * sigma + 1
        kernel_y = cv2.getGaussianKernel(ksize, sigma)
        kernel_x = kernel_y.T
        matriz_filtrada = cv2.sepFilter2D(matriz_votacion, -1, kernel_y, kernel_x)

        if debug == 1:
            matrices_votacion.append(matriz_filtrada)

        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
        z = np.unravel_index(np.argmax(matriz_filtrada, axis=None), matriz_filtrada.shape)
        q = (int(z[1] * 10), int(z[0] * 10))
        detected_points.append(q)

    return detected_points


def draw_points(images, points):
    """Dibuja un circulo en los puntos dados sobre las imagenes recibidas como parametro"""
    for index in range(len(images)):
        cv2.circle(images[index], points[index], 15, (255, 0, 0), thickness=10, lineType=8, shift=0)
        plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_RGB2BGR))
        plt.title("Imagen " + str(index + 1))
        plt.show()


def main(num_keypoints, scale_factor, pyramid_levels, knn_matches, gaussian_kernel_sigma, debug=0):
    train_images = load()
    orb = cv2.ORB_create(nfeatures=num_keypoints, scaleFactor=scale_factor, nlevels=pyramid_levels)
    match_table, flann = train(train_images, orb)
    test_images = load('test')
    test_images_color = load_color('test')
    # para hacer deteccion de una imagen en concreto, pasar esta imagen en una lista del siguiente modo
    # test_images = [test_images[i]], donde i es el indice de la imagen a testear
    detected_points = detect(test_images, orb, match_table, flann, knn_matches, gaussian_kernel_sigma, debug)
    draw_points(test_images_color, detected_points)


if __name__ == "__main__":
    NUM_KEYPOINTS = 300
    SCALE_FACTOR = 1.3
    PYRAMID_LEVELS = 4
    KNN_MATCHES = 4
    GAUSSIAN_KERNEL_SIGMA = 2
    DEBUG = 1

    # para ver las matrices de votacion, introducir el parametro DEBUG
    main(NUM_KEYPOINTS, SCALE_FACTOR, PYRAMID_LEVELS, KNN_MATCHES, GAUSSIAN_KERNEL_SIGMA)
