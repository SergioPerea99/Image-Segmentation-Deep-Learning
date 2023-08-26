import os
import cv2
import numpy as np

def quadtree(imagen, nivel, umbral):
    m, n = imagen.shape
    media = np.mean(imagen)
    desviacion_estandar = np.std(imagen)

    if m <= 2 or n <= 2 or desviacion_estandar <= umbral:
        return np.ones((m, n)) * media
    else:
        mitad_m = m // 2
        mitad_n = n // 2
        esquina_superior_izquierda = quadtree(imagen[:mitad_m, :mitad_n], nivel + 1, umbral)
        esquina_superior_derecha = quadtree(imagen[:mitad_m, mitad_n:], nivel + 1, umbral)
        esquina_inferior_izquierda = quadtree(imagen[mitad_m:, :mitad_n], nivel + 1, umbral)
        esquina_inferior_derecha = quadtree(imagen[mitad_m:, mitad_n:], nivel + 1, umbral)
        return np.vstack((np.hstack((esquina_superior_izquierda, esquina_superior_derecha)),
                          np.hstack((esquina_inferior_izquierda, esquina_inferior_derecha))))

def procesar_imagen(nombre_archivo, carpeta_entrada, carpeta_salida):
    ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
    img = cv2.imread(ruta_entrada)
    b, g, r = cv2.split(img)
    
    b_segmentado = quadtree(b, 0, 10)
    g_segmentado = quadtree(g, 0, 10)
    r_segmentado = quadtree(r, 0, 10)
    
    b_segmentado = cv2.normalize(b_segmentado, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g_segmentado = cv2.normalize(g_segmentado, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    r_segmentado = cv2.normalize(r_segmentado, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    segmentado = cv2.merge([b_segmentado, g_segmentado, r_segmentado])
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
    cv2.imwrite(ruta_salida, segmentado)

if __name__ == "__main__":
    carpeta_entrada = r"C:\TFG\1_PREPROCESAMIENTO\BBDD_1_PREPROC"
    carpeta_salida = r"C:\TFG\2_MODELOS_VC\DATASET_PRUEBAS_ETIQUETADO\1_DIVIDIR_FUSIONAR\DyV_rgb_10"
    os.makedirs(carpeta_salida, exist_ok=True)
    cont = 0

    for nombre_archivo in os.listdir(carpeta_entrada):
        procesar_imagen(nombre_archivo, carpeta_entrada, carpeta_salida)
        cont += 1
        print("Imagen segmentada número " + str(cont))

