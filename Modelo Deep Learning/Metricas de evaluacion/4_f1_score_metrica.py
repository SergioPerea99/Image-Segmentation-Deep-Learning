from sklearn.metrics import f1_score
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Inicialización de variables
carpeta_segmentaciones_modelo = r"C:\Users\Lenovo\Downloads\resultados_segmentacion\version2_modelo"
carpeta_ground_truth = r"C:\mascaras_test_originales"
f1_valores = []

# Recorrer las imágenes en las carpetas
for image_name in os.listdir(carpeta_segmentaciones_modelo):

    # Cargar las imágenes segmentada y de referencia
    img_segmentada = cv2.imread(os.path.join(carpeta_segmentaciones_modelo, image_name), cv2.IMREAD_GRAYSCALE)
    img_groundtruth = cv2.imread(os.path.join(carpeta_ground_truth, image_name), cv2.IMREAD_GRAYSCALE)


    # Crear una máscara binaria
    mascara = np.where(img_segmentada != 0, 1, 0)
    mask_ground_truth = np.where(img_groundtruth != 0, 1, 0)


    # Convertir la máscara en un vector
    vector_mascara = mascara.flatten()
    vector_ground_truth = mask_ground_truth.flatten()


    # Calcular el F1-score
    f1 = f1_score(vector_ground_truth, vector_mascara)
    f1_valores.append(f1)
    print("F1-score:", f1)

media_f1 = np.mean(f1_valores)
print("Media F1-Score: ",media_f1)