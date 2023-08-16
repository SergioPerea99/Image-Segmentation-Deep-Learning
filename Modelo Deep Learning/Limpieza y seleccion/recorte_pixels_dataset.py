import os
from PIL import Image


#1ª PARTE: Recortar las máscaras (etiquetas).

def recortar_mascaras(ruta_entrada, ruta_salida, distancia_recorte, ubicacion_recorte):
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
carpeta_entrada = r"C:\TFG\0_BBDD_ETIQUETADAS\Dataset_pruebas\imagenes_animales"
carpeta_salida = r"C:\TFG\0_BBDD_ETIQUETADAS\Dataset_pruebas\imagenes_animales_recortados"

#Información importante:
# - Primero: 225 píxeles de recorte inferior a todas.
# - Segundo: 400 píxeles de recorte en la derecha a algunas por un lado y 50 píxeles de recorte inferior en otras.
distancia_recorte = 175

ubicacion_recorte = "derecha"  # Ubicación del recorte: "inferior", "derecha", "izquierda" o "superior"


# Recorre la carpeta de entrada y procesa cada imagen encontrada
'''
cont = 0
for nombre_archivo in os.listdir(carpeta_entrada):
    cont += 1
    print(cont)
    ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
    recortar_mascaras(ruta_entrada, carpeta_salida, distancia_recorte, ubicacion_recorte)
'''



#2ª PARTE: Recortar las imágenes según las mismas dimensiones de recorte que las máscaras.


#Ahora, para las imágenes sin etiquetar. Obteniendo el mismo recorte aplicada para cada una de las imágenes.
def recortar_imagenes(ruta_entrada_origen, ruta_entrada_destino, ruta_salida, ubicacion_recorte):
    # Obtener el nombre del archivo de la ruta de entrada destino
    nombre_archivo_destino = os.path.basename(ruta_entrada_destino)
    
    # Abrir la imagen destino
    imagen_destino = Image.open(ruta_entrada_destino)
    
    # Obtener el ancho y alto de la imagen destino
    ancho_destino, alto_destino = imagen_destino.size
    
    # Abrir la imagen origen
    imagen_origen = Image.open(ruta_entrada_origen)
    
    # Obtener las dimensiones de la imagen origen
    ancho_origen, alto_origen = imagen_origen.size
    
    # Calcular las coordenadas para el recorte según la ubicación especificada y las dimensiones de la imagen origen
    if ubicacion_recorte == "inferior":
        coordenadas_recorte = (0, 0, ancho_destino, alto_destino - (alto_destino - alto_origen))
    elif ubicacion_recorte == "derecha":
        coordenadas_recorte = (0, 0, ancho_destino - (ancho_destino - ancho_origen), alto_destino)
    elif ubicacion_recorte == "izquierda":
        coordenadas_recorte = ((ancho_origen - ancho_destino), 0, ancho_destino, alto_destino)
    elif ubicacion_recorte == "superior":
        coordenadas_recorte = (0, (alto_origen - alto_destino), ancho_destino, alto_destino)
    else:
        raise ValueError("Ubicación de recorte no válida")
    
    # Realizar el recorte de la imagen destino utilizando las dimensiones de la imagen origen
    imagen_recortada = imagen_destino.crop(coordenadas_recorte)
    
    # Guardar la imagen recortada en la carpeta de salida con el mismo nombre de archivo destino
    ruta_salida_archivo = os.path.join(ruta_salida, nombre_archivo_destino)
    imagen_recortada.save(ruta_salida_archivo)


# Recorrer la carpeta de entrada de la imagen destino y procesar cada imagen encontrada
carpeta_entrada_origen = r"C:\TFG\0_BBDD_ETIQUETADAS\Dataset_etiquetado_completo_recortado_definitivo"
carpeta_entrada_destino = r"C:\TFG\0_BBDD_ETIQUETADAS\Dataset_pruebas\imagenes_animales_recortados"
carpeta_salida = r"C:\TFG\0_BBDD_ETIQUETADAS\Dataset_pruebas\imagenes_animales_recortados"


cont = 0
for nombre_archivo_destino in os.listdir(carpeta_entrada_origen):
    cont += 1
    print(cont)
    ruta_entrada_origen = os.path.join(carpeta_entrada_origen, nombre_archivo_destino)
    ruta_entrada_destino = os.path.join(carpeta_entrada_destino, nombre_archivo_destino)
    recortar_imagenes(ruta_entrada_origen, ruta_entrada_destino, carpeta_salida, ubicacion_recorte)






#3ª PARTE: Comprobar que el conjunto de imágenes y de máscaras tengan las mismas dimensiones.

def verificar_dimensiones(carpeta_origen, carpeta_destino):
    # Obtener la lista de archivos en ambas carpetas
    archivos_origen = os.listdir(carpeta_origen)
    archivos_destino = os.listdir(carpeta_destino)
    
    # Filtrar los archivos que tienen el mismo nombre en ambas carpetas
    archivos_relacionados = [archivo for archivo in archivos_origen if archivo in archivos_destino]
    
    # Verificar las dimensiones de cada par de imágenes relacionadas
    for archivo in archivos_relacionados:
        ruta_origen = os.path.join(carpeta_origen, archivo)
        ruta_destino = os.path.join(carpeta_destino, archivo)
        
        # Abrir las imágenes
        imagen_origen = Image.open(ruta_origen)
        imagen_destino = Image.open(ruta_destino)
        
        # Obtener las dimensiones de las imágenes
        ancho_origen, alto_origen = imagen_origen.size
        ancho_destino, alto_destino = imagen_destino.size
        
        # Comparar las dimensiones
        if ancho_origen != ancho_destino or alto_origen != alto_destino:
            print(f"Las dimensiones de las imágenes {archivo} no coinciden.")
    
    print("Verificación de dimensiones completada.")


verificar_dimensiones(carpeta_entrada_origen,carpeta_salida)
