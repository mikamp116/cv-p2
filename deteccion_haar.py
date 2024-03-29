import cv2 as cv


class HaarDetector:
    """Detector de regiones de interes mediante un clasificador en cascada."""
    def __init__(self, classifier_file, scale_factor=1.3, min_neighbors=5):
        """Crea y entrena un detector en cascada"""
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

    def detect(self, images, scale_factor=None, min_neighbors=None):
        """Devuelve una lista con la coordenada x e y de la esquina superior derecha, la anchura y la altura de la
        region de interes detectada para cada una de las imagenes"""
        if scale_factor is None:
            scale_factor = self.scale_factor
        if min_neighbors is None:
            min_neighbors = self.min_neighbors
        return [self.classifier.detectMultiScale(image, scale_factor, min_neighbors) for image in images]
