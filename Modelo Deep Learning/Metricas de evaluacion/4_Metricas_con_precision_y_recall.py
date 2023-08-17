import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image
import os
import math

target_color = [246,246,246]  # The RGB color for segmentation

# Inicialización de variables
segmented_folder = r"C:\Users\Lenovo\Downloads\resultados_segmentacion"
reference_folder = r"C:\mascaras_test_originales"
f1_valores = []
boundary_f1_score_valores = []


#Métodos de interés

def obtener_coordenadas_color_contornos(imagen_grises_resaltada, color_objetivo):
    
    # Encontrar los contornos de los objetos resaltados
    contornos, _ = cv2.findContours(imagen_grises_resaltada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Obtener las coordenadas (x, y) de todos los puntos del contorno
    coordenadas_x = []
    coordenadas_y = []
    for cnt in contornos:
        for punto in cnt:
            x, y = punto[0]
            coordenadas_x.append(int(x))
            coordenadas_y.append(int(y))

    return coordenadas_x, coordenadas_y

def mostrar_puntos_de_interes(imagen, coordenadas_x, coordenadas_y, color):

    # Crear una máscara vacía
    mascara = np.zeros(imagen.shape[:2], dtype=np.uint8)
    
    # Dibujar los puntos de interés en la máscara
    for x, y in zip(coordenadas_x, coordenadas_y):
        cv2.circle(mascara, (x, y), 5, color, -1)
    
    # Aplicar la máscara a la imagen original
    imagen_con_puntos = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_con_puntos = cv2.bitwise_and(imagen_con_puntos, imagen_con_puntos, mask=mascara)
    
    return imagen_con_puntos

# Recorrer las imágenes en las carpetas
for image_name in os.listdir(segmented_folder):

    # Cargar las imágenes segmentada y de referencia
    img_segmented = cv2.imread(os.path.join(segmented_folder, image_name))
    img_reference = cv2.imread(os.path.join(reference_folder, image_name))

    # Carga la imagen
    image = Image.open(os.path.join(segmented_folder, image_name))

    try:
        img_reference_gray = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
        img_segmented_gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)
    except:
        continue


    # 2. MÉTRICA F1-SCORE (ACCURACY Y RECALL POR IGUAL):

    # Crear una máscara binaria del objeto segmentado en la imagen de referencia
    object_mask = (img_reference_gray != 0)

    # Calcular el número de píxeles coincidentes entre el objeto segmentado y la imagen segmentada
    TP_matching_pixels_object = np.sum(np.logical_and(object_mask, img_reference_gray == img_segmented_gray)) #TP bien

    #Ahora calculamos el número de píxeles que se han segmentado con el RGB pero que en realidad no pertenecen a la clase de la imagen de referencia
    target_pixels_segmented = np.all(img_segmented == target_color, axis=-1)
    FP_target_pixels = np.sum(np.logical_and(target_pixels_segmented, img_reference_gray == 0)) #FP bien
    FN_target_pixels = np.sum(np.logical_and(~target_pixels_segmented, img_reference_gray != 0))
    

    # Calcular el Accuracy y el Recall de los píxeles en el objeto segmentado
    pixel_accuracy_object =TP_matching_pixels_object / (TP_matching_pixels_object + FP_target_pixels) 
    pixel_recall_object = TP_matching_pixels_object / (TP_matching_pixels_object + FN_target_pixels) 


    #Calcular el F1-Score a partir del accuracy y el recall
    f1_score = (2 * pixel_accuracy_object * pixel_recall_object) / (pixel_accuracy_object + pixel_recall_object)
    if not math.isnan(f1_score):
        f1_valores.append(f1_score * 100)


    coordenadas_x, coordenadas_y = obtener_coordenadas_color_contornos(img_reference_gray, target_color)
    img_boundary_reference = mostrar_puntos_de_interes(img_reference,coordenadas_x,coordenadas_y,(37, 177, 90))

    coordenadas_x, coordenadas_y = obtener_coordenadas_color_contornos(img_segmented_gray, target_color)
    img_boundary_segmented = mostrar_puntos_de_interes(img_segmented,coordenadas_x,coordenadas_y,(37, 177, 90))

    try:
        img_boundary_reference_gray = cv2.cvtColor(img_boundary_reference, cv2.COLOR_BGR2GRAY)
        img_boundary_segmented_gray = cv2.cvtColor(img_boundary_segmented, cv2.COLOR_BGR2GRAY)
    except:
        continue

    object_boundary_mask = (img_boundary_reference_gray != 0)

    TP_matching_pixels_object = np.sum(np.logical_and(object_boundary_mask, img_boundary_reference_gray == img_boundary_segmented_gray)) #TP
    target_pixels_segmented = np.all(img_boundary_segmented == target_color, axis=-1)
    FP_target_pixels = np.sum(np.logical_and(target_pixels_segmented, img_boundary_reference_gray == 0))
    FN_target_pixels = np.sum(np.logical_and(~target_pixels_segmented, img_boundary_reference_gray != 0))

    # Calcular el Accuracy y el Recall de los píxeles en el objeto segmentado
    pixel_accuracy_object =TP_matching_pixels_object / (TP_matching_pixels_object + FP_target_pixels)
    pixel_recall_object = TP_matching_pixels_object / (TP_matching_pixels_object + FN_target_pixels)
    boundary_F1_SCORE = (2 * pixel_accuracy_object * pixel_recall_object) / (pixel_accuracy_object + pixel_recall_object)
    if not math.isnan(boundary_F1_SCORE):
        boundary_f1_score_valores.append(boundary_F1_SCORE * 100)



media_f1 =  np.mean(f1_valores)
print("Media de F1 Score: ",media_f1)
media_boundary_f1 =  np.mean(boundary_f1_score_valores)
print("Media de F1 Score: ",media_boundary_f1)