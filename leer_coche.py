import string
import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import localizacion_caracteres as localizacion
from reconocimiento_caracteres import ReconocimientoCaracteres


def get_name(path):
    if path.find('/') != -1:
        filename = path[path.find('/'):]
    elif path.find('\\') != -1:
        filename = path[path.find('\\'):]
    else:
        filename = path
    return filename


def leer(directorio, visualizar):
    input_images_color, nombre_imagenes = localizacion.load(directorio, color=True)

    filename = get_name(directorio) + '.txt'
    file = open(filename, 'w')

    numbers = [str(n) for n in list(range(10))]
    letters = string.ascii_uppercase
    valid_letters = [c for c in
                     letters.replace('A', '').replace('E', '').replace('I', '').replace('O', '').replace('U', '')]

    numeros = ReconocimientoCaracteres(numbers, 'training_ocr')
    C, E = numeros.entrenar()
    # numeros.test(C)

    letras = ReconocimientoCaracteres(valid_letters, 'training_ocr')
    C, E = letras.entrenar()
    # letras.test(C)

    ### DETECCION ###

    local, matriculas, caracteres = localizacion.localizar(directorio)

    local_resized = []
    for image_list in local:
        aux = []
        for image in image_list:
            aux.append(
                [cv.resize(char, (10, 10), 0, 0, cv.INTER_LINEAR).reshape(1, 100).astype(np.float64) for char in image])
        local_resized.append(aux)

    for img in range(len(local_resized)):
        imagen = input_images_color[img]
        for m in range(len(local_resized[img])):
            F = np.array([char for char in local_resized[img][m]])[:, 0, :]
            out_numeros = numeros.reconocer(F[:4])
            out_letras = [letras.included_characters[i] for i in letras.reconocer(F[-3:])]
            texto_matricula_numeros = str(out_numeros[0]) + str(out_numeros[1]) + str(out_numeros[2]) \
                                      + str(out_numeros[3])
            texto_matricula_letras = str(out_letras[0]) + str(out_letras[1]) + str(out_letras[2])

            (x, y, w, h) = matriculas[img][m]

            nombre_imagen = get_name(nombre_imagenes[img])
            x_centro_matricula = x + w // 2
            y_centro_matricula = y + h // 2
            texto_matricula = texto_matricula_numeros + ' ' + texto_matricula_letras
            mitad_largo_matricula = w // 2
            file.write(str(nombre_imagen) + ' ' + str(x_centro_matricula) + ' ' + str(y_centro_matricula) + ' ' +
                       texto_matricula + ' ' + str(mitad_largo_matricula) + '\n')

            imagen = cv.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
            imagen = cv.circle(imagen, (x_centro_matricula, y_centro_matricula), 5, (0, 255, 0), thickness=2,
                               lineType=8, shift=0)

            matricula = caracteres[img][m]
            caracteres_matricula = texto_matricula.replace(' ', '')
            for i in range(len(matricula)):
                (a, b, c, d) = matricula[i]
                imagen = cv.rectangle(imagen, (x+a, y+b), (x+a + c, y+b + d), (0, 0, 255), 1)
                imagen = cv.putText(imagen, caracteres_matricula[i], (x+a,y+b), cv.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 1, cv.LINE_AA)

        if visualizar:
            plt.imshow(imagen)
            plt.show()

    file.close()


if __name__ == "__main__":
    assert len(sys.argv) > 1, \
        "Debes introducir al menos un argumento, el directorio donde se encuentran las imagenes.\n" \
        "Uso: python leer_coche.py <path_completo_directorio_coches> [<visualizar_resultados>] \n" \
        "El parametro entre corchetes es opcional y su valor por defecto es 'False'."

    if len(sys.argv) > 2:
        visualizar_resultados = sys.argv[2]
    else:
        visualizar_resultados = False
    directorio_imagenes = sys.argv[1]

    leer(directorio_imagenes, visualizar_resultados)
