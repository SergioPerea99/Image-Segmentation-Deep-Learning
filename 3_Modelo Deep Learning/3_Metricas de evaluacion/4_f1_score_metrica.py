from sklearn.metrics import f1_score
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os



#Métodos necesarios para el Boundary f1 score

def obtener_coordenadas_color_contornos(imagen_grises_resaltada):
    
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

    # Dibujar los puntos de interés en la máscara
    mascara = np.zeros_like(imagen, dtype=np.uint8)
    imagen_con_puntos = imagen.copy()
    for x, y in zip(coordenadas_x, coordenadas_y):
        cv2.circle(mascara, (x, y), 5, color, -1)
    
    # Aplicar la máscara a la imagen original
    imagen_con_puntos = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_con_puntos = cv2.bitwise_and(imagen_con_puntos, imagen_con_puntos, mask=mascara)

    return imagen_con_puntos


# Inicialización de variables
carpeta_segmentaciones_modelo = r"C:\Users\Lenovo\Downloads\resultados_segmentacion\version8_modelo"
carpeta_ground_truth = r"C:\mascaras_test_originales"
f1_valores = []
boundary_f1_valores = []
max_f1_valor = -1.0
max_f1_nombre_archivo = ""
min_f1_valor = 1.0
min_f1_nombre_archivo = ""

# Recorrer las imágenes en las carpetas
for image_name in os.listdir(carpeta_segmentaciones_modelo):
    try:

        # F1 Score
        img_segmentada = cv2.imread(os.path.join(carpeta_segmentaciones_modelo, image_name), cv2.IMREAD_GRAYSCALE)
        img_groundtruth = cv2.imread(os.path.join(carpeta_ground_truth, image_name), cv2.IMREAD_GRAYSCALE)


        mascara = np.where(img_segmentada != 0, 1, 0)
        mask_ground_truth = np.where(img_groundtruth != 0, 1, 0)


        vector_mascara = mascara.flatten()
        vector_ground_truth = mask_ground_truth.flatten()


        f1 = f1_score(vector_ground_truth, vector_mascara)

        if f1 > max_f1_valor:
            max_f1_valor = f1
            max_f1_nombre_archivo = image_name
        
        if f1 < min_f1_valor:
            min_f1_valor = f1
            min_f1_nombre_archivo = image_name

        f1_valores.append(f1)
        print("F1-score:", f1)
    except:
        print(image_name)
        continue


    #Boundary F1 Score
    try:
        coordenadas_x, coordenadas_y = obtener_coordenadas_color_contornos(img_groundtruth)
        img_boundary_reference = mostrar_puntos_de_interes(img_groundtruth,coordenadas_x,coordenadas_y,(249,249,249))

        coordenadas_x, coordenadas_y = obtener_coordenadas_color_contornos(img_segmentada)
        img_boundary_segmented = mostrar_puntos_de_interes(img_segmentada,coordenadas_x,coordenadas_y,(249,249,249))

        img_boundary_reference = cv2.cvtColor(img_boundary_reference, cv2.COLOR_BGR2GRAY)
        img_boundary_segmented = cv2.cvtColor(img_boundary_segmented,cv2.COLOR_BGR2GRAY)


        boundary_mascara_segmentada = np.where(img_boundary_segmented != 0, 1, 0)
        boundary_mascara_ground_truth = np.where(img_boundary_reference != 0, 1, 0)

        boundary_vector_mascara_segmentada = boundary_mascara_segmentada.flatten()
        boundary_vector_ground_truth = boundary_mascara_ground_truth.flatten()

        boundary_f1 = f1_score(boundary_vector_ground_truth, boundary_vector_mascara_segmentada)
        boundary_f1_valores.append(boundary_f1)
        print("Boundary F1-score: ",boundary_f1)
    except:
        print(image_name)
        continue


media_f1 = np.mean(f1_valores)
media_boundary_f1 = np.mean(boundary_f1_valores)
print("Media F1-Score: ",media_f1)
print("Media Boundary F1-Score: ",media_boundary_f1)

print("Minimo valor de F1-Score: ",min_f1_valor," - Imagen: ",min_f1_nombre_archivo)
print("Maximo valor de F1-Score: ",max_f1_valor," - Imagen: ",max_f1_nombre_archivo)

# Definir colores según los valores de F1-score
colors = ['red' if score < 0.5 else 'blue' for score in f1_valores]

# Valores para los ejes x e y (en este caso, serán los mismos valores)
x_values = f1_valores
y_values = boundary_f1_valores

# Crear la gráfica de puntos
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, color=colors, marker='o')
plt.title('Gráfico de Puntos de F1-score')
plt.xlabel('F1-score')
plt.ylabel('Boundary F1-score')
plt.xlim(0, 1)
plt.ylim(0, 0.2)
plt.grid(True)
plt.show()