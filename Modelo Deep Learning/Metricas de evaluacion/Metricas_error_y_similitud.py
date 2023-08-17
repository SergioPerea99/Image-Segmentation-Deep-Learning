import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os

# Inicialización de variables
segmented_folder = r"C:\Users\Lenovo\Downloads\resultados_segmentacion"
reference_folder = r"C:\mascaras_test_originales"
mae_values = []
mse_values = []
ssim_values = []


# Recorrer las imágenes en las carpetas
for image_name in os.listdir(segmented_folder):

    # Cargar las imágenes segmentada y de referencia
    img_segmented = cv2.imread(os.path.join(segmented_folder, image_name))
    img_reference = cv2.imread(os.path.join(reference_folder, image_name))
    
    #Ejecutar el cálculo MAE, MSE y SSIM entre ambas.
    img_reference_gray = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
    img_segmented_gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)

    plt.imshow(img_segmented_gray)
    plt.show()

    plt.imshow(img_reference_gray)
    plt.show()

    
    mae = np.mean(np.abs(img_reference_gray - img_segmented_gray))
    mse = np.mean((img_reference_gray - img_segmented_gray) ** 2)
    ssim_valor = ssim(img_reference_gray, img_segmented_gray)
    
    print(image_name,": ",mae,mse,ssim_valor)

    # Añadir cada una a su lista correspondiente
    mae_values.append(mae)
    mse_values.append(mse)
    ssim_values.append(ssim_valor)
        

# Calcular las medias de las métricas e imrpimirlas
mean_mae = np.mean(mae_values)
mean_mse = np.mean(mse_values)
mean_ssim = np.mean(ssim_values)


print("Mean MAE:", mean_mae)
print("Mean MSE:", mean_mse)
print("Mean SSIM:", mean_ssim)



