import cv2 as cv

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html


class HaarDetector:
    def __init__(self, classifier_file, scale_factor=1.3, min_neighbors=5):
        self.classifier = cv.CascadeClassifier(classifier_file)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def get_scale_factor(self):
        return self.scale_factor

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

    def get_min_neighbors(self):
        return self.min_neighbors

    def set_min_neighbors(self, min_neighbors):
        self.min_neighbors = min_neighbors

    def detect(self, images):
        """Devuelve una lista con la coordenada x e y de la esquina superior derecha, la anchura y la altura de la
        region de interes detectada (roi) para cada una de las imagenes"""
        return [self.classifier.detectMultiScale(image, self.scale_factor, self.min_neighbors) for image in images]

    def detect2(self, images, scale_factor, min_neighbors):
        """Devuelve una lista con la coordenada x e y de la esquina superior derecha, la anchura y la altura de la
        region de interes detectada (roi) para cada una de las imagenes"""
        return [self.classifier.detectMultiScale(image, scale_factor, min_neighbors) for image in images]
