import os
from PIL import Image


#1ª PARTE: Recortar las máscaras (etiquetas).
def recortar_mascaras(ruta_entrada, ruta_salida, distancia_recorte, ubicacion_recorte):
    # Cargar la imagen
    nombre_archivo = os.path.basename(ruta_entrada)
    imagen = Image.open(ruta_entrada)
    ancho, alto = imagen.size
    
    # Realizar el recorte de la imagen:
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
    
    imagen_recortada = imagen.crop(coordenadas_recorte)
    
    # Guardar la imagen
    ruta_salida_archivo = os.path.join(ruta_salida, nombre_archivo)
    imagen_recortada.save(ruta_salida_archivo)




#2ª PARTE: Recortar las imágenes según las mismas dimensiones de recorte que las máscaras.
def recortar_imagenes(ruta_entrada_origen, ruta_entrada_destino, ruta_salida, ubicacion_recorte):
    
    # Cargar las imagenes
    nombre_archivo_destino = os.path.basename(ruta_entrada_destino)
    imagen_destino = Image.open(ruta_entrada_destino)
    ancho_destino, alto_destino = imagen_destino.size
    imagen_origen = Image.open(ruta_entrada_origen)
    ancho_origen, alto_origen = imagen_origen.size
    
    # Realizar el recorte de la imagen destino utilizando las dimensiones de la imagen origen
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
    
    imagen_recortada = imagen_destino.crop(coordenadas_recorte)
    
    # Guardar la imagen
    ruta_salida_archivo = os.path.join(ruta_salida, nombre_archivo_destino)
    imagen_recortada.save(ruta_salida_archivo)


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


if __name__ == "__main__":

    carpeta_entrada_origen = r"C:\Users\Lenovo\Downloads\Dataset_pruebas\Masks\val"
    carpeta_entrada_destino = r"C:\Users\Lenovo\Downloads\Dataset_pruebas\Images\val"
    
    verificar_dimensiones(carpeta_entrada_origen,carpeta_entrada_destino)
