import localizacion_caracteres as localizacion
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


def area(elem):
    return elem[2] * elem[3]


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


training_ocr = localizacion.load('training_ocr', False, ['.', 'A', 'E', 'I', 'O', 'U'])
training_ocr_color = localizacion.load('training_ocr', True, ['.', 'A', 'E', 'I', 'O', 'U'])
# training_ocr_umbralizado = localizacion.umbralizado(training_ocr, 9, 2)
training_ocr_umbralizado = localizacion.umbralizado(training_ocr, True, 2)
# for (umbral, color) in zip(training_ocr_umbralizado, training_ocr_color):
#     plt.imshow(umbral, "gray")
#     plt.show()
#     plt.imshow(color, "gray")
#     plt.show()

traning_ocr_caracteres = get_contornos_caracteres(training_ocr_umbralizado)

contador = np.zeros((31), dtype=np.uint16)

# for i in range(len(training_ocr)):  # para cada imagen
#     image = training_ocr[i]
#     color = training_ocr_color[i]
#     caracteres = traning_ocr_caracteres[i]
#     caracteres.sort(key=area, reverse=True)
#     for (x, y, w, h) in caracteres[:2]:
#         color = cv.rectangle(color, (x, y), (x + w, y + h), (0, 0, 255), 1)
#         # print(np.mean(image[y:y + h, x:x + w]))
#         # print(np.std(image[y:y + h, x:x + w]))
#         # print("---")
#     contador[i // 250] += len(caracteres[:2])
#     # plt.imshow(color)
#     # plt.show()


# for (image, contour) in zip(training_ocr_color, traning_ocr_caracteres):
#     for (x, y, w, h) in contour:
#         image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         print(np.mean(image[:,:,1]))
#         print(np.std(image[:,:,1]))
#         print("---")
#     plt.imshow(image)
#     plt.show()

roi_training_ocr = []
for i in range(len(training_ocr)):  # para cada imagen
    image = training_ocr[i]
    caracteres = traning_ocr_caracteres[i]
    caracteres.sort(key=area, reverse=True)
    for (x, y, w, h) in caracteres[:2]:
        roi_training_ocr.append(image[y:y + h, x:x + w])
    contador[i // 250] += len(caracteres[:2])

# for image in roi_training_ocr:
#     plt.imshow(image)
#     plt.show()

# contador = [c.astype(np.float32) for c in contador]

E = [i for i in range(contador.shape[0]) for _ in range(contador[i])]

# E = []
# for i in range(contador.shape[0]):
#     for _ in range(contador[i]):
#         E.append(i)

# roi_training_ocr = [image[y:y + h, x:x + w] for (image, contour) in zip(training_ocr, traning_ocr_caracteres)
#                     for (x, y, w, h) in contour]

traning_ocr_resized = [cv.resize(image, (10, 10), 0, 0, cv.INTER_LINEAR) for image in roi_training_ocr]

C = [char.reshape(1, 100).astype(np.float64) for char in traning_ocr_resized]
C2 = np.array([char.reshape(1, 100).astype(np.float64) for char in traning_ocr_resized])
C3 = C2[:,0,:]
#E = [float(e) for e in E]

lda = LinearDiscriminantAnalysis()
lda.fit(C3,E)
CR = lda.transform(C3)

gnb = GaussianNB()
gnb.fit(CR, E)
output = gnb.predict(CR)

pass
