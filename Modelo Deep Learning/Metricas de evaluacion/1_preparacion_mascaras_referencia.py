import os
import numpy as np
from PIL import Image



def preparar_imagenes_originales(ruta_imagen, rango_rgb_inicio=[35, 175, 88],rango_rgb_fin=[39, 179, 92],  rgb_convertido=[37, 177, 90]):
    imagen = Image.open(ruta_imagen)
    imagen_np = np.array(imagen)

    # Define los valores RGB de inicio y fin del rango
    rango_rgb_inicio = np.array(rango_rgb_inicio)
    rango_rgb_fin = np.array(rango_rgb_fin)

    # Crea una máscara basada en el rango de valores RGB
    mascara = np.all((imagen_np >= rango_rgb_inicio) & (imagen_np <= rango_rgb_fin), axis=-1)

    # Crea una imagen con los valores de RGB (37, 177, 90)
    imagen_resultante = np.zeros_like(imagen_np) 
    imagen_resultante[mascara] = imagen_np[mascara]

    # Convierte el arreglo NumPy de vuelta a una imagen PIL
    imagen_resultante_pil = Image.fromarray(imagen_resultante.astype(np.uint8))

    return imagen_resultante_pil

carpeta_imagenes = r"C:\Users\Lenovo\Downloads\Dataset_pruebas\Masks\test"
carpeta_destino = r"C:\mascaras_test_originales"
os.makedirs(carpeta_destino, exist_ok=True)

for imagen in os.listdir(carpeta_imagenes):
    ruta_imagen_original = os.path.join(carpeta_imagenes, imagen)
    imagen_final = preparar_imagenes_originales(ruta_imagen_original)

    # Guardar la imagen resultante en la carpeta de resultados
    ruta_imagen_resultante = os.path.join(carpeta_destino, imagen)

    imagen_final.save(ruta_imagen_resultante)