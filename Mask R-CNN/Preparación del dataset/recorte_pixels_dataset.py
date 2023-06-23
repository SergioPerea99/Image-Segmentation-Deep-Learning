import os
from PIL import Image

def recortar_imagen(ruta_entrada, ruta_salida, distancia_recorte, ubicacion_recorte):
    # Obtiene el nombre del archivo de la ruta de entrada
    nombre_archivo = os.path.basename(ruta_entrada)
    
    # Abre la imagen
    imagen = Image.open(ruta_entrada)
    
    # Obtiene el ancho y alto de la imagen
    ancho, alto = imagen.size
    
    # Calcula las coordenadas para el recorte según la ubicación especificada
    if ubicacion_recorte == "inferior":
        coordenadas_recorte = (0, 0, ancho, alto - distancia_recorte)
    elif ubicacion_recorte == "derecha":
        coordenadas_recorte = (0, 0, ancho - distancia_recorte, alto)
    elif ubicacion_recorte == "izquierda":
        coordenadas_recorte = (distancia_recorte, 0, ancho, alto)
    elif ubicacion_recorte == "superior":
        coordenadas_recorte = (0, distancia_recorte, ancho, alto)
    else:
        raise ValueError("Ubicación de recorte no válida")
    
    # Realiza el recorte de la imagen
    imagen_recortada = imagen.crop(coordenadas_recorte)
    
    # Guarda la imagen recortada en la carpeta de salida con el mismo nombre de archivo
    ruta_salida_archivo = os.path.join(ruta_salida, nombre_archivo)
    imagen_recortada.save(ruta_salida_archivo)

# Rutas de la carpeta de entrada y la carpeta de salida
carpeta_entrada = r"C:\TFG\0_BBDD_ETIQUETADAS\Dataset_pruebas\Images\train"
carpeta_salida = r"C:\TFG\0_BBDD_ETIQUETADAS\Dataset_pruebas\Images\train"

#Información importante:
# - Primero: 225 píxeles de recorte inferior a todas.
# - Segundo: 400 píxeles de recorte en la derecha a algunas por un lado y 50 píxeles de recorte inferior en otras.
distancia_recorte = 225

ubicacion_recorte = "inferior"  # Ubicación del recorte: "inferior", "derecha", "izquierda" o "superior"


# Recorre la carpeta de entrada y procesa cada imagen encontrada
cont = 0
for nombre_archivo in os.listdir(carpeta_entrada):
    cont += 1
    print(cont)
    ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
    recortar_imagen(ruta_entrada, carpeta_salida, distancia_recorte, ubicacion_recorte)
