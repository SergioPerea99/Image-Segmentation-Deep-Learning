import os
import cv2
import numpy as np

# Definir la ruta a la carpeta de sentrada y salida
carpeta_entrada = 'C:\EJEMPLOS_BBDD\BASE'
carpeta_salida = 'C:\EJEMPLOS_BBDD\K_MEANS_4'

# Crear la carpeta de salida si no existe
os.makedirs(carpeta_salida, exist_ok=True)
cont = 0

# Enumerar todos los archivos en la carpeta de entrada
for nombre_archivo in os.listdir(carpeta_entrada):
    # Construir la ruta completa al archivo de entrada
    ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
    
    # Cargar la imagen en color
    img = cv2.imread(ruta_entrada)
    
    # Preprocesar la imagen
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a formato RGB
    img_aplanada = img.reshape((-1, 3)).astype(np.float32)  # Aplanar y convertir a tipo float32
    
    # Aplicar el algoritmo de k-means
    k = 4  # Número de clusters
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1.0)  # Criterio de parada
    _, etiquetas, centros = cv2.kmeans(img_aplanada, k, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS) #Criterio de inicialización de centroides
    
    # Reasignar los colores originales a los clusters
    imagen_segmentada = centros[etiquetas.flatten().astype(np.uint8)].reshape(img.shape)
    
    # Guardar la imagen segmentada
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
    cv2.imwrite(ruta_salida, imagen_segmentada)
    
    cont += 1
    print("Imagen segmentada número " + str(cont))
