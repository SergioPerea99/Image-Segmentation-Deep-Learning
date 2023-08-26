import os
import cv2
import numpy as np

def cargar_imagen(ruta):
    return cv2.imread(ruta)

def convertir_a_escala_grises(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def aplicar_umbral(img_gris, umbral_min, umbral_max):
    _, img_binaria = cv2.threshold(img_gris, umbral_min, umbral_max, cv2.THRESH_TOZERO)
    return img_binaria

def filtrar_contornos(contornos, umbral_area):
    contornos_filtrados = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > umbral_area:
            contornos_filtrados.append(contorno)
    return contornos_filtrados

def crear_mascara(img, contornos):
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contornos, -1, (255, 255, 255), thickness=cv2.FILLED)
    return mask

def aplicar_mascara(img, mask):
    return cv2.bitwise_and(img, mask)

if __name__ == "__main__":
    carpeta_entrada = r"C:\TFG\0_BBDD_COMPLETAS\BBDD_completa\Primer enlace\Lynx_pardinus\Babel"
    carpeta_salida = r"C:\TFG\2_MODELOS_VC\DATASET_PRUEBAS_ETIQUETADO\4_CONTORNOS\contornos_180_255_THRESH_TOZERO_INV_INTERESANTE_60_200_THRESH_TOZERO"
    umbral_area = 60

    os.makedirs(carpeta_salida, exist_ok=True)
    cont = 0

    for nombre_archivo in os.listdir(carpeta_entrada):
        ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
        img = cargar_imagen(ruta_entrada)
        
        img_gris = convertir_a_escala_grises(img)
        img_binaria = aplicar_umbral(img_gris, 60, 200)
        
        contornos, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_filtrados = filtrar_contornos(contornos, umbral_area)
        
        mask = crear_mascara(img, contornos_filtrados)
        
        img_segmentada = aplicar_mascara(img, mask)
        
        ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
        cv2.imwrite(ruta_salida, img_segmentada)
        
        cont += 1
        print("Imagen segmentada número " + str(cont))
