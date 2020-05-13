import numpy
import string
import cv2
from matplotlib import pyplot
import os
import sys
from reconocimiento_caracteres import ReconocimientoCaracteres


def leer():
    numbers = [str(n) for n in list(range(10))]
    letters = string.ascii_uppercase
    valid_letters = [c for c in
                     letters.replace('A', '').replace('E', '').replace('I', '').replace('O', '').replace('U', '')]

    numeros = ReconocimientoCaracteres(numbers, 'training_ocr')

    C, E = numeros.entrenar()
    numeros.test(C, E)


    letras = ReconocimientoCaracteres(valid_letters, 'training_ocr')
    C, E = letras.entrenar()
    letras.test(C, E)


if __name__ == "__main__":
    leer()
