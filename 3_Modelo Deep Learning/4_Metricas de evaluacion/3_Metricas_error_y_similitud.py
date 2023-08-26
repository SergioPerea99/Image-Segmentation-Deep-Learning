import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def cargar_imagenes(segmented_folder, reference_folder):
    segmented_images = []
    reference_images = []

    for image_name in os.listdir(segmented_folder):
        img_segmented = cv2.imread(os.path.join(segmented_folder, image_name))
        img_reference = cv2.imread(os.path.join(reference_folder, image_name))
        
        if img_segmented is not None and img_reference is not None:
            segmented_images.append(img_segmented)
            reference_images.append(img_reference)

    return segmented_images, reference_images

def calcular_metricas(segmented_images, reference_images):
    mae_values = []
    mse_values = []
    ssim_values = []

    for img_segmented, img_reference in zip(segmented_images, reference_images):
        img_reference_gray = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
        img_segmented_gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)

        mae = np.mean(np.abs(img_reference_gray - img_segmented_gray))
        mse = np.mean((img_reference_gray - img_segmented_gray) ** 2)
        ssim_value = ssim(img_reference_gray, img_segmented_gray)

        mae_values.append(mae)
        mse_values.append(mse)
        ssim_values.append(ssim_value)

    return mae_values, mse_values, ssim_values

def mostrar_metricas(mae_values, mse_values, ssim_values):
    mean_mae = np.mean(mae_values)
    mean_mse = np.mean(mse_values)
    mean_ssim = np.mean(ssim_values)

    print("Mean MAE:", mean_mae)
    print("Mean MSE:", mean_mse)
    print("Mean SSIM:", mean_ssim)

if __name__ == "__main__":
    segmented_folder = r"C:\Users\Lenovo\Downloads\resultados_segmentacion\version8_modelo"
    reference_folder = r"C:\mascaras_test_originales"
    
    segmented_images, reference_images = cargar_imagenes(segmented_folder, reference_folder)
    mae_values, mse_values, ssim_values = calcular_metricas(segmented_images, reference_images)
    mostrar_metricas(mae_values, mse_values, ssim_values)
