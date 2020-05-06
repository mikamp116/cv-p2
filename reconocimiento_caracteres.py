import string
import numpy as np
import cv2 as cv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import localizacion_caracteres as localizacion


def area(rectangulo):
    """Devuelve el area de un rectangulo"""
    return rectangulo[2] * rectangulo[3]


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


numbers = [str(n) for n in list(range(10))]
letters = string.ascii_uppercase
valid_letters = [c for c in
                 letters.replace('A', '').replace('E', '').replace('I', '').replace('O', '').replace('U', '')]

training_ocr = localizacion.load('training_ocr', numbers, False)
# training_ocr_umbralizado = localizacion.umbralizado(training_ocr, 9, 2)
training_ocr_umbralizado = localizacion.umbralizado(training_ocr, True, 2)

traning_ocr_caracteres = get_contornos_caracteres(training_ocr_umbralizado)

contador = np.zeros((len(numbers)), dtype=np.uint16)

roi_training_ocr = []
for i in range(len(training_ocr)):  # para cada imagen
    image = training_ocr[i]
    caracteres = traning_ocr_caracteres[i]
    caracteres.sort(key=area, reverse=True)
    # A
    # for (x, y, w, h) in caracteres[:2]:
    #     roi_training_ocr.append(image[y:y + h, x:x + w])
    # contador[i // 250] += len(caracteres[:2])
    # B
    if (len(caracteres)>0):
        p = 0
        a = caracteres[p]
        while (area(a) > 0.75 * image.shape[0]*image.shape[1] and p < len(caracteres)-1):
            p+= 1
            a = caracteres[p]
        roi_training_ocr.append(image[a[1]:a[1] + a[3], a[0]:a[0] + a[2]])
        contador[i // 250] += 1


E = [i for i in range(contador.shape[0]) for _ in range(contador[i])]


traning_ocr_resized = [cv.resize(image, (10, 10), 0, 0, cv.INTER_LINEAR) for image in roi_training_ocr]

C = [char.reshape(1, 100).astype(np.float64) for char in traning_ocr_resized]
C2 = np.array([char.reshape(1, 100).astype(np.float64) for char in traning_ocr_resized])
C3 = C2[:, 0, :]

lda = LinearDiscriminantAnalysis()
lda.fit(C3, E)
CR = lda.transform(C3)

gnb = GaussianNB()
gnb.fit(CR, E)
output = gnb.predict(CR)

acierto = 0
for i in output:
    if i // 250 == int(output[i]):
        acierto += 1
acierto = acierto / len(output)
print(acierto)

pass
