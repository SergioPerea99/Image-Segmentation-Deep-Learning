import os
import cv2
import numpy as np



def procesar_imagen(nombre_archivo, carpeta_entrada):
    ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
    img = cv2.imread(ruta_entrada)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a formato RGB
    img_aplanada = img.reshape((-1, 3)).astype(np.float32)  # Aplanar y convertir a tipo float32
    return img.shape, img_aplanada

#Método de K-Means
def k_means(img_aplanada,img_shape,k,max_iteraciones,iteraciones_inicializacion_centroides):
    # Aplicar el algoritmo de k-means
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iteraciones, 1.0)  # Criterio de parada
    _, etiquetas, centros = cv2.kmeans(img_aplanada, k, None, criterio, iteraciones_inicializacion_centroides, cv2.KMEANS_RANDOM_CENTERS) #Criterio de inicialización de centroides
    
    # Reasignar los colores originales a los clusters
    imagen_segmentada = centros[etiquetas.flatten().astype(np.uint8)].reshape(img_shape)

    return imagen_segmentada



if __name__ == "__main__":
    # Inicialización de variables
    carpeta_entrada =  r"C:\TFG\1_PREPROCESAMIENTO\BBDD_1_PREPROC"
    carpeta_salida = r"C:\TFG\2_MODELOS_VC\DATASET_PRUEBAS_ETIQUETADO\5_K_MEANS\k4_200_50"
    os.makedirs(carpeta_salida, exist_ok=True)
    cont = 0
    k = 4  # Número de clusters

    # Enumerar todos los archivos en la carpeta de entrada
    for nombre_archivo in os.listdir(carpeta_entrada):
        
        # Cargar la imagen
        img_shape, img_aplanada = procesar_imagen(nombre_archivo, carpeta_entrada)
                        
        #Algoritmo K-Means
        imagen_segmentada = k_means(img_aplanada,img_shape,k,200,50)

        # Guardar la imagen 
        ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
        cv2.imwrite(ruta_salida, imagen_segmentada)
        cont += 1
        print("Imagen segmentada número " + str(cont))
