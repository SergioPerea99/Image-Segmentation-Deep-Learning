import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def obtener_coordenadas_color_rango(ruta_imagen, rango_color, tolerancia=10):
    # Leer la imagen
    imagen = cv2.imdecode(np.fromfile(ruta_imagen, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    # Convertir los rangos de color a arreglos numpy
    rango_color = np.array(rango_color, dtype=np.uint8)

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    mascara = cv2.inRange(imagen_rgb, rango_color, rango_color)

    # Encontrar los puntos con el color en el rango especificado
    puntos_color = np.argwhere(mascara > 0)

    if tolerancia > 0:
        puntos_reducidos = []
        puntos_reducidos.append(puntos_color[0])
        for i in range(1, len(puntos_color)):
            punto_actual = puntos_color[i]
            punto_anterior = puntos_reducidos[-1]
            distancia = np.linalg.norm(punto_actual - punto_anterior)
            if distancia >= tolerancia:
                puntos_reducidos.append(punto_actual)
        puntos_color = np.array(puntos_reducidos)
    
    # Obtener las coordenadas x e y de los puntos
    coordenadas_x = puntos_color[:, 1].tolist()
    coordenadas_y = puntos_color[:, 0].tolist()
    
    return coordenadas_x, coordenadas_y


def obtener_json_segmentacion(carpeta_imagenes, carpeta_jsons, color_segmentacion):
    # Lista para almacenar los datos de cada imagen
    datos_imagenes = []
    cont_modif = 1
    # Recorrer la carpeta de imágenes y subcarpetas
    for root, dirs, files in os.walk(carpeta_imagenes):
        for filename in files:
            # Obtener la ruta completa del archivo
            ruta_imagen = os.path.join(root, filename)
            ruta_json_imagen = os.path.join(carpeta_jsons, filename)

            # Verificar la extensión del archivo
            extension = os.path.splitext(filename)[1].lower()
            if extension == '.jpg' or extension == '.png' or extension == '.JPG':
                # Obtener coordenadas de color en la imagen
                coordenadas_x, coordenadas_y = obtener_coordenadas_color_rango(ruta_imagen, color_segmentacion)
                
                # Verificar que se encontraron coordenadas
                if coordenadas_x and coordenadas_y:

                    # Reemplazar caracteres especiales en el nombre de archivo
                    nuevo_nombre_archivo = filename
                    modificacion = False
                    if "ñ" in nuevo_nombre_archivo or "Ñ" in nuevo_nombre_archivo:
                        modificacion = True
                        print(nuevo_nombre_archivo)
                        nuevo_nombre_archivo = nuevo_nombre_archivo.replace("ñ", "n").replace("Ñ","N")
                    if "ó" in nuevo_nombre_archivo or "Ó" in nuevo_nombre_archivo:
                        modificacion = True
                        print(nuevo_nombre_archivo)
                        nuevo_nombre_archivo = nuevo_nombre_archivo.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("Ó","O")


                    # Leer la imagen
                    imagen = cv2.imdecode(np.fromfile(ruta_imagen, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

                    # Obtener el tamaño de la imagen en bytes
                    size = len(cv2.imencode('.' + extension, imagen)[1].tobytes())

                    # Crear el diccionario de regiones
                    regiones = {
                        '0': {
                            'shape_attributes': {
                                'name': 'polygon',
                                'all_points_x': list(coordenadas_x),
                                'all_points_y': list(coordenadas_y)
                            },
                            'region_attributes': {}
                        }
                    }

                    # Crear el diccionario principal para la imagen actual
                    json_data = {
                        nuevo_nombre_archivo + str(size):{
                            'fileref': '',
                            'size': size,
                            'filename': nuevo_nombre_archivo,
                            'base64_img_data': '',
                            'file_attributes': {},
                            'regions': regiones
                        }
                    }

                    # Agregar el diccionario de la imagen actual a la lista
                    datos_imagenes.append(json_data)

                    print(len(datos_imagenes))

                    # Renombrar la imagen
                    if modificacion:
                        # Obtener la ruta de la imagen con el nuevo nombre
                        nuevo_ruta = os.path.join(root, nuevo_nombre_archivo)
                        nuevo_ruta_json = os.path.join(carpeta_jsons, nuevo_nombre_archivo)
                        print("Modificacion numero ",cont_modif)
                        os.rename(ruta_imagen, nuevo_ruta)
                        os.rename(ruta_json_imagen, nuevo_ruta_json)
                    cont_modif += 1
                else:
                    print(f"No se encontraron coordenadas en la imagen: {ruta_imagen}")
            else:
                print(f"El archivo no es una imagen: {ruta_imagen}")

    return datos_imagenes
    

# Carpeta que contiene las imágenes y subcarpetas
carpeta_imagenes = r"Dataset_pruebas/Masks/val"
carpeta_jsons = r"Dataset_pruebas/Images/val"

# Rangos de color para buscar en las imágenes
color_segmentacion = (37, 177, 90)

# Especificar la ruta y el nombre de archivo donde se guardará el JSON
archivo_json = r"Dataset_pruebas/Images/val/puntos_de_interes.json"

# Obtener los datos de segmentación en formato JSON para todas las imágenes en la carpeta
datos_imagenes = obtener_json_segmentacion(carpeta_imagenes, carpeta_jsons, color_segmentacion)

# Guardar los datos de las imágenes en el archivo JSON
with open(archivo_json, "w") as archivo:
    json.dump(datos_imagenes, archivo, ensure_ascii=False)

# FIXEAMOS el JSON CREADO de forma que quede como el ejemplo para poder ser usado en MASK-RCNN
import json

# Especificar la ruta y el nombre de archivo donde se guardará el JSON
archivo_json = r"Dataset_pruebas/Images/val/puntos_de_interes.json"

print("PRIMER ARCHIVO COMPLETADO: ", str(archivo_json))

# Abrir el archivo JSON en modo lectura
with open(archivo_json, "r") as archivo:
    # Leer el contenido del archivo y almacenarlo en una variable
    contenido_json = archivo.read()

# Realizar las modificaciones necesarias en la cadena de texto
contenido_modificado = contenido_json.replace("}, {", ",")
contenido_modificado = contenido_modificado.replace("ñ", "n").replace("Ñ","N")
contenido_modificado = contenido_modificado.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
contenido_modificado = contenido_modificado[1:-1]  # IMPORTANTE: PARA MANTENER LA MISMA ESTRUCTURA, HAY QUE ELIMINAR EL PRIMER Y ÚLTIMO CARACTER DEL ARCHIVO FIXEADO.

# Guardar la cadena de texto modificada en el mismo archivo o en un nuevo archivo
# Guardar la cadena de texto modificada en un nuevo archivo
nuevo_archivo_json = r"Dataset_pruebas/Images/val/puntos_de_interes_nuevo.json"
with open(nuevo_archivo_json, "w") as archivo:
    archivo.write(contenido_modificado)

print("SEGUNDO ARCHIVO COMPLETADO: ", str(nuevo_archivo_json))
