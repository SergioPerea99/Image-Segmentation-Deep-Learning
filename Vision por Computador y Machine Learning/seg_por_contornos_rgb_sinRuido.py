import os
import cv2
import numpy as np

# Definir la ruta a la carpeta de entrada y salida
carpeta_entrada = 'C:\TFG\DATASET_COMPLETO_ETIQUETADO\\4_CONTORNOS\seg_contornos_180_255_THRESH_TOZERO_INV_INTERESANTE'
carpeta_salida = 'C:\TFG\DATASET_COMPLETO_ETIQUETADO\\8_CONTORNOS_CONTORNOS_INVERSOS_sinRuido\contornos_180_255_THRESH_TOZERO_INV_INTERESANTE_60_200_THRESH_TOZERO'

umbral_area = 60 #Píxeles de umbral de área para limpieza de los que quedan sueltos.

# Crear la carpeta de salida si no existe
os.makedirs(carpeta_salida, exist_ok=True)
cont = 0

# Enumerar todos los archivos en la carpeta de entrada
for nombre_archivo in os.listdir(carpeta_entrada):
    # Construir la ruta completa al archivo de entrada
    ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
    
    # Cargar la imagen en color
    img = cv2.imread(ruta_entrada)
    
    # Convertir la imagen a escala de grises
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un umbral para obtener una imagen binaria
    _, img_binaria = cv2.threshold(img_gris, 60, 200, cv2.THRESH_TOZERO)
    
    # Encontrar los contornos en la imagen binaria
    contornos, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar los contornos para eliminar los que estén sueltos
    contornos_filtrados = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > umbral_area:  # Establecer un umbral de área mínimo para considerar un contorno como válido
            contornos_filtrados.append(contorno)

    # Crear una máscara en blanco del mismo tamaño que la imagen original
    mask = np.zeros_like(img)

    # Dibujar los contornos filtrados en la máscara
    cv2.drawContours(mask, contornos_filtrados, -1, (255, 255, 255), thickness=cv2.FILLED)

    
    # Aplicar la máscara a la imagen original para obtener la imagen segmentada
    img_segmentada = cv2.bitwise_and(img, mask)
    
    # Construir la ruta completa al archivo de salida
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
    
    # Guardar la imagen segmentada
    cv2.imwrite(ruta_salida, img_segmentada)
    
    cont += 1
    print("Imagen segmentada número " + str(cont))
