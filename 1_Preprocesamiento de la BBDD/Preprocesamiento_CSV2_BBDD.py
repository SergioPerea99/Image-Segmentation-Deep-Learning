import os
import cv2 as cv
from PIL import Image


def leer_csv(nombre_csv):
    diccionario = {}
    nombres_registros_dicc = []
    especies = set()
    with open(nombre_csv, encoding="utf8") as csv_file:
        lines = csv_file.readlines()
        header = lines[0].strip().split(';')
        indice_especie = header.index(especie_columna)
        indice_nombre_img = header.index(nombre_img_columna)
        for line in lines[1:]:
            data = line.strip().split(';')
            key = data[indice_nombre_img]
            value = data[indice_especie]
            diccionario[key] = value
            nombres_registros_dicc.append(key)
            especies.add(value)
    return diccionario

def procesar_imagenes(diccionario, ruta, ruta_copiar_animales, lista_no_animales):
    no_aparecen_total = 0
    no_aparecen = 0
    descarte = 0
    animals = 0
    imagenes_encontradas_total = 0

    for root, dirs, files in os.walk(ruta, topdown=False):
        for name in files:
            imagenes_encontradas_total += 1
            lista_imagenes.append(name)

            if diccionario.get(name):
                valor = diccionario.get(name)

                if valor not in lista_no_animales:
                    ruta_img = os.path.join(root, name)
                    img = cv.imread(ruta_img)
                    im_with_exif = Image.open(ruta_img)
                    pil_img = Image.fromarray(img)
                    animals += 1

                    ruta_copia = os.path.join(ruta_copiar_animales, name)
                    if 'exif' in im_with_exif.info:
                        exif = im_with_exif.info['exif']
                        pil_img.save(ruta_copia, format='JPEG', exif=exif)
                    else:
                        pil_img.save(ruta_copia, format='JPEG')

                else:
                    descarte += 1

            else:
                no_aparecen += 1
                no_aparecen_total += 1

        # Mostrar estadísticas por directorio
        print("\n")
        print(os.path.basename(root))
        print("Número de imágenes encontradas:", len(files))
        print("Número de imágenes que no aparecen en el archivo CSV:", no_aparecen)
        print("Número de imágenes que contienen animales:", animals)

        # Reiniciar variables
        animals = 0

    # Mostrar estadísticas globales
    print("Número total de imágenes:", imagenes_encontradas_total)
    print("Número de imágenes descartadas:", descarte)


#Ejecución
if __name__ == "__main__":
    especie_columna = "Species"
    nombre_img_columna = "FileName"
    lista_no_animales = ["human", "NOID", "vehicle", "NA", "", "empty"]
    nombre_csv = "CSV2.csv"
    ruta = r"C:\TFG\0_BBDD_COMPLETAS\BBDD_completa\Segundo enlace"
    ruta_copiar_animales = r"C:\TFG\1_PREPROCESAMIENTO\BBDD_2_PREPROC"
    lista_imagenes = []

    diccionario = leer_csv(nombre_csv)
    procesar_imagenes(diccionario, ruta, ruta_copiar_animales, lista_no_animales)
