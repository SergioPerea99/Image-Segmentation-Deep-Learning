import os
import random
import shutil

def dividir_imagenes(carpeta_origen, carpeta_destino):
    # Crear carpetas de entrenamiento, validación y prueba
    carpeta_entrenamiento = os.path.join(carpeta_destino, "train")
    carpeta_validacion = os.path.join(carpeta_destino, "val")
    carpeta_prueba = os.path.join(carpeta_destino, "test")
    os.makedirs(carpeta_entrenamiento, exist_ok=True)
    os.makedirs(carpeta_validacion, exist_ok=True)
    os.makedirs(carpeta_prueba, exist_ok=True)

    # Obtener la lista de imágenes en la carpeta de origen
    imagenes = [archivo for archivo in os.listdir(carpeta_origen)]
    total_imagenes = len(imagenes)
    random.shuffle(imagenes)

    # Calcular la cantidad de imágenes para cada conjunto
    tamanio_entrenamiento = int(0.7 * total_imagenes)
    print(tamanio_entrenamiento)
    tamanio_validacion = int(0.15 * total_imagenes)
    print(tamanio_validacion)

    # Mover las imágenes a las carpetas correspondientes
    for i, imagen in enumerate(imagenes):
        ruta_origen = os.path.join(carpeta_origen, imagen)
        if i < tamanio_entrenamiento:
            ruta_destino = os.path.join(carpeta_entrenamiento, imagen)
        elif i < tamanio_entrenamiento + tamanio_validacion:
            ruta_destino = os.path.join(carpeta_validacion, imagen)
        else:
            ruta_destino = os.path.join(carpeta_prueba, imagen)
        shutil.copy2(ruta_origen, ruta_destino)

    print("División de imágenes completada.")



def copiar_imagenes(carpeta_fuente, carpeta_destino_entrenamiento, carpeta_destino_validacion, carpeta_destino_prueba):
    # Rutas de las carpetas train, val y test
    carpeta_imagenes = r"C:\TFG\0_BBDD_ETIQUETADAS\Dataset_pruebas\Images"
    carpeta_entrenamiento = os.path.join(carpeta_imagenes, "train")
    carpeta_validacion = os.path.join(carpeta_imagenes, "val")
    carpeta_prueba = os.path.join(carpeta_imagenes, "test")

    # Lista de nombres de archivos sin extensión
    archivos_entrenamiento = [os.path.splitext(archivo)[0] for archivo in os.listdir(carpeta_entrenamiento)]
    archivos_validacion = [os.path.splitext(archivo)[0] for archivo in os.listdir(carpeta_validacion)]
    archivos_prueba = [os.path.splitext(archivo)[0] for archivo in os.listdir(carpeta_prueba)]

    # Recorrer todas las imágenes en la carpeta principal
    for ruta_actual, _, archivos in os.walk(carpeta_fuente):
        for archivo in archivos:
            nombre_archivo = os.path.splitext(archivo)[0]
            ruta_imagen = os.path.join(ruta_actual, archivo)
            if nombre_archivo in archivos_entrenamiento:
                ruta_destino = os.path.join(carpeta_destino_entrenamiento, archivo)
            elif nombre_archivo in archivos_validacion:
                ruta_destino = os.path.join(carpeta_destino_validacion, archivo)
            elif nombre_archivo in archivos_prueba:
                ruta_destino = os.path.join(carpeta_destino_prueba, archivo)
            else:
                #print(f"No se encontró la imagen {archivo} en las carpetas entrenamiento, validacion o prueba.")
                continue

            shutil.copy2(ruta_imagen, ruta_destino)
            print(f"La imagen {archivo} se copió correctamente en {ruta_destino}")



# Ruta de la carpeta que contiene las imágenes originales
carpeta_origen = r"C:\TFG\0_BBDD_ETIQUETADAS\TODAS_cuidado_hay_duplicados"
# Ruta de la carpeta donde se guardarán las imágenes divididas
carpeta_destino = r"C:\TFG\0_BBDD_ETIQUETADAS\Dataset_pruebas\Masks"

# Llamar a la función para dividir las imágenes
dividir_imagenes(carpeta_origen, carpeta_destino)


# Rutas de las carpetas de destino según la carpeta en la que se encuentre la imagen
carpeta_destino_entrenamiento = r"C:\TFG\0_BBDD_ETIQUETADAS\Dataset_pruebas\Masks\train"
carpeta_destino_validacion = r"C:\TFG\0_BBDD_ETIQUETADAS\Dataset_pruebas\Masks\val"
carpeta_destino_prueba = r"C:\TFG\0_BBDD_ETIQUETADAS\Dataset_pruebas\Masks\test"

# Llamar a la función para buscar y copiar las imágenes
copiar_imagenes(carpeta_origen, carpeta_destino_entrenamiento, carpeta_destino_validacion, carpeta_destino_prueba)
