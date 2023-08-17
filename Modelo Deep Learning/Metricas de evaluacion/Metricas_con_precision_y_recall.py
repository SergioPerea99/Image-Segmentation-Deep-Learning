import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os


target_color = [37, 177, 90]  # The RGB color for segmentation

# Inicialización de variables
segmented_folder = r"C:\Users\Lenovo\Downloads\resultados_segmentacion"
reference_folder = r"C:\mascaras_test_originales"
f1_valores = []

# Recorrer las imágenes en las carpetas
for image_name in os.listdir(segmented_folder):

    # Cargar las imágenes segmentada y de referencia
    img_segmented = cv2.imread(os.path.join(segmented_folder, image_name))
    img_reference = cv2.imread(os.path.join(reference_folder, image_name))

    img_segmented_rgb = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB)
    img_reference_rgb = cv2.cvtColor(img_reference, cv2.COLOR_BGR2RGB)
    
    #Ejecutar el cálculo MAE, MSE y SSIM entre ambas.
    img_reference_gray = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
    img_segmented_gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)

    plt.imshow(img_segmented_gray)
    plt.show()

    plt.imshow(img_reference_gray)
    plt.show()


    # 2. MÉTRICA F1-SCORE (ACCURACY Y RECALL POR IGUAL):

    # Crear una máscara binaria del objeto segmentado en la imagen de referencia
    object_mask = (img_reference_gray != 0)

    # Calcular el número de píxeles coincidentes entre el objeto segmentado y la imagen segmentada
    TP_matching_pixels_object = np.sum(np.logical_and(object_mask, img_reference_gray == img_segmented_gray)) #TP

    #Ahora calculamos el número de píxeles que se han segmentado con el RGB (37, 177, 90) pero que en realidad no pertenecen a la clase de la imagen de referencia
    target_pixels_segmented = np.all(img_segmented == target_color, axis=-1)
    FP_target_pixels = np.sum(np.logical_and(target_pixels_segmented, img_reference_gray == 0))
    FN_target_pixels = np.sum(np.logical_and(~target_pixels_segmented, img_reference_gray != 0))

    print(TP_matching_pixels_object, FP_target_pixels, FN_target_pixels)

    # Calcular el Accuracy y el Recall de los píxeles en el objeto segmentado
    pixel_accuracy_object =TP_matching_pixels_object / (TP_matching_pixels_object + FP_target_pixels)
    pixel_recall_object = TP_matching_pixels_object / (TP_matching_pixels_object + FN_target_pixels)
    print("Pixel Accuracy (Object Segmented):", pixel_accuracy_object)
    print("Pixel Recall (Object Segmented):", pixel_recall_object)


    #Calcular el F1-Score a partir del accuracy y el recall
    F1_SCORE = (2 * pixel_accuracy_object * pixel_recall_object) / (pixel_accuracy_object + pixel_recall_object)
    print("F1-SCORE (Object Segmented):", F1_SCORE)
    f1_valores.append(F1_SCORE)

media_f1 = np.mean(F1_SCORE)
print("Media de F1 Score: ",media_f1)

