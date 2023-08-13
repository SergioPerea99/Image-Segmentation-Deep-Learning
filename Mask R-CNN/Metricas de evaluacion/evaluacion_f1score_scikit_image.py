
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

'''
target_color = [37, 177, 90]  # The RGB color for segmentation

# Cargar las imágenes
img_reference = cv2.imread(r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Metricas de evaluacion\output_image_reference.png")
img_segmented = cv2.imread(r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Metricas de evaluacion\output_image.png")

# Convertir a escala de grises
img_reference_gray = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
img_segmented_gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)

# 1. MÉTRICAS BÁSICAS (MAE, MSE, SSIM):


# Calcular MAE (Mean Absolute Error)
mae = np.mean(np.abs(img_reference_gray - img_segmented_gray))

# Calcular MSE (Mean Squared Error)
mse = np.mean((img_reference_gray - img_segmented_gray) ** 2)

# Calcular SSIM (Structural Similarity Index)
ssim_valor = ssim(img_reference_gray, img_segmented_gray)

print("MAE:", mae)
print("MSE:", mse)
print("SSIM:", ssim_valor)



# 2. MÉTRICA F1-SCORE (ACCURACY Y RECALL POR IGUAL):

# Crear una máscara binaria del objeto segmentado en la imagen de referencia
object_mask = (img_reference_gray != 0)

# Calcular el número de píxeles coincidentes entre el objeto segmentado y la imagen segmentada
TP_matching_pixels_object = np.sum(np.logical_and(object_mask, img_reference_gray == img_segmented_gray)) #TP

#Ahora calculamos el número de píxeles que se han segmentado con el RGB (37, 177, 90) pero que en realidad no pertenecen a la clase de la imagen de referencia
target_pixels_segmented = np.all(img_segmented == target_color, axis=-1)
FP_target_pixels = np.sum(np.logical_and(target_pixels_segmented, img_reference_gray == 0))
FN_target_pixels = np.sum(np.logical_and(~target_pixels_segmented, img_reference_gray != 0))


# Calcular el Accuracy y el Recall de los píxeles en el objeto segmentado
pixel_accuracy_object =TP_matching_pixels_object / (TP_matching_pixels_object + FP_target_pixels)
pixel_recall_object = TP_matching_pixels_object / (TP_matching_pixels_object + FN_target_pixels)
print("Pixel Accuracy (Object Segmented):", pixel_accuracy_object)
print("Pixel Recall (Object Segmented):", pixel_recall_object)


#Calcular el F1-Score a partir del accuracy y el recall
F1_SCORE = (2 * pixel_accuracy_object * pixel_recall_object) / (pixel_accuracy_object + pixel_recall_object)
print("F1-SCORE (Object Segmented):", F1_SCORE)


# 3. MÉTRICA DE BOUNDARY F1-SCORE (Cálculo de cuán bien clasifica el contorno del objeto):

'''

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
    
    # Mostrar la imagen resultante
    plt.imshow(imagen_con_puntos)
    plt.axis('off')
    plt.show()
    return imagen_con_puntos

'''
coordenadas_x, coordenadas_y = obtener_coordenadas_color_contornos(img_reference_gray, target_color)
img_boundary_reference = mostrar_puntos_de_interes(img_reference,coordenadas_x,coordenadas_y,(37, 177, 90))

coordenadas_x, coordenadas_y = obtener_coordenadas_color_contornos(img_segmented_gray, target_color)
img_boundary_segmented = mostrar_puntos_de_interes(img_segmented,coordenadas_x,coordenadas_y,(37, 177, 90))

#Ya tenemos las imágenes con únicamente los contornos, por lo que se procede al cálculo de TP,FP Y FN.

img_boundary_reference_gray = cv2.cvtColor(img_boundary_reference, cv2.COLOR_BGR2GRAY)
img_boundary_segmented_gray = cv2.cvtColor(img_boundary_segmented, cv2.COLOR_BGR2GRAY)

object_boundary_mask = (img_boundary_reference_gray != 0)

TP_matching_pixels_object = np.sum(np.logical_and(object_boundary_mask, img_boundary_reference_gray == img_boundary_segmented_gray)) #TP
target_pixels_segmented = np.all(img_boundary_segmented == target_color, axis=-1)
FP_target_pixels = np.sum(np.logical_and(target_pixels_segmented, img_boundary_reference_gray == 0))
FN_target_pixels = np.sum(np.logical_and(~target_pixels_segmented, img_boundary_reference_gray != 0))

# Calcular el Accuracy y el Recall de los píxeles en el objeto segmentado
pixel_accuracy_object =TP_matching_pixels_object / (TP_matching_pixels_object + FP_target_pixels)
pixel_recall_object = TP_matching_pixels_object / (TP_matching_pixels_object + FN_target_pixels)
print("Pixel Boundary Accuracy (Object Segmented):", pixel_accuracy_object)
print("Pixel Boundary Recall (Object Segmented):", pixel_recall_object)
F1_SCORE = (2 * pixel_accuracy_object * pixel_recall_object) / (pixel_accuracy_object + pixel_recall_object)
print("BOUNDARY F1-SCORE (Object Segmented):", F1_SCORE)




# 4. Métrica de Jaccard Index

# Definir el color RGB objetivo (37, 177, 90)
target_color = [37, 177, 90]

# Crear máscaras binarias para las regiones segmentadas y de referencia
mask_reference = np.all(img_reference == target_color, axis=-1)
mask_segmented = np.all(img_segmented == target_color, axis=-1)

# Calcular la intersección de las máscaras
intersection = np.logical_and(mask_reference, mask_segmented)

# Calcular la unión de las máscaras
union = np.logical_or(mask_reference, mask_segmented)

# Calcular el Índice de Jaccard (IoU)
iou = np.sum(intersection) / np.sum(union)

print("Jaccard Index (IoU):", iou)

'''


import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

# Definir la carpeta con las imágenes segmentadas y las imágenes de referencia
segmented_folder = r"C:\Users\Lenovo\Downloads\Dataset_pruebas\Images\test"
reference_folder = r"C:\Users\Lenovo\Downloads\Dataset_pruebas\Masks\test"

# Definir el color RGB objetivo (37, 177, 90)
target_color = [37, 177, 90]

# Crear listas para almacenar las métricas calculadas
mae_values = []
mse_values = []
ssim_values = []
f1_score_values = []
boundary_f1_score_values = []
jaccard_index_values = []

# Recorrer las imágenes en las carpetas
for image_name in os.listdir(segmented_folder):
    # Cargar las imágenes segmentada y de referencia
    img_segmented = cv2.imread(os.path.join(segmented_folder, image_name))
    img_reference = cv2.imread(os.path.join(reference_folder, image_name))

    img_reference_gray = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
    img_segmented_gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)

    mae = np.mean(np.abs(img_reference_gray - img_segmented_gray))
    mse = np.mean((img_reference_gray - img_segmented_gray) ** 2)
    ssim_valor = ssim(img_reference_gray, img_segmented_gray)


    object_mask = (img_reference_gray != 0)

    # Calcular el número de píxeles coincidentes entre el objeto segmentado y la imagen segmentada
    TP_matching_pixels_object = np.sum(np.logical_and(object_mask, img_reference_gray == img_segmented_gray)) #TP

    #Ahora calculamos el número de píxeles que se han segmentado con el RGB (37, 177, 90) pero que en realidad no pertenecen a la clase de la imagen de referencia
    target_pixels_segmented = np.all(img_segmented == target_color, axis=-1)
    FP_target_pixels = np.sum(np.logical_and(target_pixels_segmented, img_reference_gray == 0))
    FN_target_pixels = np.sum(np.logical_and(~target_pixels_segmented, img_reference_gray != 0))


    # Calcular el Accuracy y el Recall de los píxeles en el objeto segmentado
    pixel_accuracy_object =TP_matching_pixels_object / (TP_matching_pixels_object + FP_target_pixels)
    pixel_recall_object = TP_matching_pixels_object / (TP_matching_pixels_object + FN_target_pixels)
    F1_SCORE = (2 * pixel_accuracy_object * pixel_recall_object) / (pixel_accuracy_object + pixel_recall_object)
    
    coordenadas_x, coordenadas_y = obtener_coordenadas_color_contornos(img_reference_gray, target_color)
    img_boundary_reference = mostrar_puntos_de_interes(img_reference,coordenadas_x,coordenadas_y,(37, 177, 90))

    coordenadas_x, coordenadas_y = obtener_coordenadas_color_contornos(img_segmented_gray, target_color)
    img_boundary_segmented = mostrar_puntos_de_interes(img_segmented,coordenadas_x,coordenadas_y,(37, 177, 90))

    #Ya tenemos las imágenes con únicamente los contornos, por lo que se procede al cálculo de TP,FP Y FN.

    img_boundary_reference_gray = cv2.cvtColor(img_boundary_reference, cv2.COLOR_BGR2GRAY)
    img_boundary_segmented_gray = cv2.cvtColor(img_boundary_segmented, cv2.COLOR_BGR2GRAY)

    object_boundary_mask = (img_boundary_reference_gray != 0)

    TP_matching_pixels_object = np.sum(np.logical_and(object_boundary_mask, img_boundary_reference_gray == img_boundary_segmented_gray)) #TP
    target_pixels_segmented = np.all(img_boundary_segmented == target_color, axis=-1)
    FP_target_pixels = np.sum(np.logical_and(target_pixels_segmented, img_boundary_reference_gray == 0))
    FN_target_pixels = np.sum(np.logical_and(~target_pixels_segmented, img_boundary_reference_gray != 0))

    # Calcular el Accuracy y el Recall de los píxeles en el objeto segmentado
    pixel_accuracy_object =TP_matching_pixels_object / (TP_matching_pixels_object + FP_target_pixels)
    pixel_recall_object = TP_matching_pixels_object / (TP_matching_pixels_object + FN_target_pixels)
    boundary_F1_SCORE = (2 * pixel_accuracy_object * pixel_recall_object) / (pixel_accuracy_object + pixel_recall_object)+


    # Crear máscaras binarias para las regiones segmentadas y de referencia
    mask_reference = np.all(img_reference == target_color, axis=-1)
    mask_segmented = np.all(img_segmented == target_color, axis=-1)

    # Calcular la intersección de las máscaras
    intersection = np.logical_and(mask_reference, mask_segmented)

    # Calcular la unión de las máscaras
    union = np.logical_or(mask_reference, mask_segmented)

    # Calcular el Índice de Jaccard (IoU)
    iou = np.sum(intersection) / np.sum(union)

    # Agregar las métricas calculadas a las listas correspondientes
    mae_values.append(mae)
    mse_values.append(mse)
    ssim_values.append(ssim_valor)
    f1_score_values.append(F1_SCORE)
    boundary_f1_score_values.append(boundary_F1_SCORE)
    jaccard_index_values.append(iou)

# Calcular las medias de las métricas
mean_mae = np.mean(mae_values)
mean_mse = np.mean(mse_values)
mean_ssim = np.mean(ssim_values)
mean_f1_score = np.mean(f1_score_values)
mean_boundary_f1_score = np.mean(boundary_f1_score_values)
mean_jaccard_index = np.mean(jaccard_index_values)

# Imprimir las medias de las métricas
print("Mean MAE:", mean_mae)
print("Mean MSE:", mean_mse)
print("Mean SSIM:", mean_ssim)
print("Mean F1-SCORE:", mean_f1_score)
print("Mean BOUNDARY F1-SCORE:", mean_boundary_f1_score)
print("Mean Jaccard Index (IoU):", mean_jaccard_index)
