# Computer Vision - Practica 2: Segmentación y reconocimiento de caracteres (OCR)

## 0. Demo



## 1. Localización de los caracteres de la matricula

Este apartado consiste en la construcción de un sistema que permita la localización de los dígitos de la matrícula de un coche
dada la caja que contiene el frontal del coche. Para ello, se parte de la detección del frontal del coche realizada en 
[cv-p1](https://github.com/mikamp116/cv-p1), y en esta sección se realizará el siguiente paso necesario para la lectura de matrículas.

Para la construcción del sistema de segmentación de los dígitos se usará un enfoque basado en la umbralización de la imagen y en la detección
de componentes conexas.

## 2. Reconocimiento de los caracteres localizados (OCR)

Este apartado consiste en la construcción de un sistema que clasifique correctamente los caracteres de una matrícula.

Para ello se proporciona un conjunto de entrenamiento que contiene 250 variaciones de cada uno de los caracteres que puede aparecer 
en una matrícula (en concreto los caracteres 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ y adicionalmente el símbolo con la
E de España y la bandera de la UE que aparece a la izquierda de las matrículas actuales).

## 3. Integración de todos los algoritmos (localizar coche, matrícula y OCR)

Este apartado consiste en desarrollar un sistema que utilice los detectores desarrollados en [cv-p1](https://github.com/mikamp116/cv-p1) y en este 
repositorio para construir un programa que lea la matricula de los coches que aparezcan en imágenes y vídeo.

## 4. Ejecución

La práctica deberá ejecutarse sobre Python 3.7.X y OpenCV 4.3 mediante el siguiente mandato:

<code>>$ python leer_coche.py [directorio de imágenes de testing] [visualización de imágenes por pantalla]</code>
, donde los argumentos son los siguientes:

-   *directorio de imágenes de testing*: Ruta relativa del directorio que contiene las imágenes de testing.

-   *visualización de imágenes por pantalla*: Puede tomar los valores *True* o *False*.

## 5. Implementación y descripción del código

Véase el documento *memoria.ipynb*
