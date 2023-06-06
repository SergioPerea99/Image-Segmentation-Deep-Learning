import os
import cv2
import numpy as np

def quadtree(imagen, nivel, umbral):
    """Función recursiva para implementar el algoritmo de dividir y fusionar."""

    m, n = imagen.shape
    media = np.mean(imagen)
    desviacion_estandar = np.std(imagen)

    # Si la imagen es lo suficientemente pequeña, o si la desviación estándar está por debajo del umbral, devuelve la media de la imagen
    if m <= 2 or n <= 2 or desviacion_estandar <= umbral:
        return np.ones((m, n)) * media

    # Divide la imagen en 4 regiones y aplica la función recursivamente a cada una
    else:
        mitad_m = m // 2
        mitad_n = n // 2
        esquina_superior_izquierda = quadtree(imagen[:mitad_m, :mitad_n], nivel + 1, umbral)
        esquina_superior_derecha = quadtree(imagen[:mitad_m, mitad_n:], nivel + 1, umbral)
        esquina_inferior_izquierda = quadtree(imagen[mitad_m:, :mitad_n], nivel + 1, umbral)
        esquina_inferior_derecha = quadtree(imagen[mitad_m:, mitad_n:], nivel + 1, umbral)

        # Combina las regiones en una sola imagen
        return np.vstack((np.hstack((esquina_superior_izquierda, esquina_superior_derecha)), np.hstack((esquina_inferior_izquierda, esquina_inferior_derecha))))


# Definir la ruta a la carpeta de entrada y salida
carpeta_entrada = 'C:\EJEMPLOS_BBDD\BASE'
carpeta_salida = 'C:\EJEMPLOS_BBDD\DyF'

# Crear la carpeta de salida si no existe
os.makedirs(carpeta_salida, exist_ok=True)
cont = 0

# Enumerar todos los archivos en la carpeta de entrada
for nombre_archivo in os.listdir(carpeta_entrada):
    # Construir la ruta completa al archivo de entrada
    ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)

    # Cargar la imagen en color
    img = cv2.imread(ruta_entrada)

    # Separar los canales de color
    b, g, r = cv2.split(img)

    # Aplicar la segmentación a cada canal de color
    b_segmentado = quadtree(b, 0, 15)
    g_segmentado = quadtree(g, 0, 15)
    r_segmentado = quadtree(r, 0, 15)

    # Normalizar cada canal al rango 0-255
    b_segmentado = cv2.normalize(b_segmentado, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g_segmentado = cv2.normalize(g_segmentado, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    r_segmentado = cv2.normalize(r_segmentado, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Combinar los canales de color en una sola imagen
    segmentado = cv2.merge([b_segmentado, g_segmentado, r_segmentado])

    # Construir la ruta completa al archivo de salida
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)

    # Guardar la imagen resultante
    cv2.imwrite(ruta_salida, segmentado)
    cont += 1
    print("Imagen segmentada número " + str(cont))
