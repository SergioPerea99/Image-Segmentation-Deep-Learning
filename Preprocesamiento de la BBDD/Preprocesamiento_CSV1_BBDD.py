# Imports
import csv
import os
import cv2 as cv
from PIL import Image

# Configuración
RUTA = 'C:\TFG\Preprocesamiento\BBDD_completa\Primer enlace'
NOMBRE_CSV = "CSV1.csv"
RUTA_COPIAR_ANIMALES = ".\\BBDDPost\\prueba\\"
dimensiones = {} #Para saber las diferentes dimensiones existentes dentro del dataset


# Obtención de un diccionario {nombreArchivo : especie,animal} a partir del fichero CSV
def crear_diccionario(csv_path):
    diccionario = {}
    with open(csv_path, encoding="utf8") as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        cabecera = next(reader)
        indice_animal = cabecera.index("HierarchicalSubject")
        indice_especie = cabecera.index("Specie")
        indice_nombre_img = cabecera.index("FileName")

        for fila in reader:
            key = fila[indice_nombre_img]
            value = [fila[indice_especie], fila[indice_animal]]
            diccionario[key] = value

    return diccionario

# Procesar y copiar imágenes
def procesar_imagenes(ruta_origen, ruta_destino):
    contador = 0
    contador_resize = 0
    imagenes_encontradas = 0
    for root, dirs, files in os.walk(ruta_origen, topdown=False):
        for name in files:
            contador += 1
            imagenes_encontradas += 1
            ruta_img = os.path.join(root, name)
            try:
                img = cv.imread(ruta_img)
                #pil_img = Image.fromarray(img)
                #exif = Image.open(ruta_img).info.get('exif', b'')
                #pil_img.save(os.path.join(ruta_destino, name), format='JPEG', exif=exif)
                
                # Agregar dimensiones al diccionario
                dimension = img.shape[:2]  # Obtiene alto y ancho
                if dimension not in dimensiones:
                    dimensiones[dimension] = 0
                dimensiones[dimension] += 1
          
            except Exception as e:
                contador_resize += 1
                
        print(f"\nLa carpeta llamada {os.path.basename(root)} tiene {imagenes_encontradas} imágenes.")
        imagenes_encontradas = 0
    print(f"\nEl total de imágenes es de {contador}")


# Ejecución
def main():
    diccionario = crear_diccionario(NOMBRE_CSV)
    print(f"Líneas leídas: {len(diccionario)}")
    procesar_imagenes(RUTA, RUTA_COPIAR_ANIMALES)
    print("Dimensiones de las imágenes y sus conteos: ")
    for dimension, count in dimensiones.items():
        print(f"Dimensión: {dimension}, contador: {count}")





## HILO PRINCIPAL
main()
