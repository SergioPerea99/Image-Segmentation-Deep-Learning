import os
import os.path
import cv2 as cv
from PIL import Image

# Variables globales
ruta = 'C:\TFG\BBDD_1_2\BBDD_completa\Segundo enlace'
listaCarpetas = ['\L1','\L2','\L3','\L4','\L5','\L6','\L7','\L8','\L9',r'\N1',r'\N2',r'\N3',r'\N4',r'\N5',r'\N6',r'\N7',r'\N8',r'\N9']

nombreCSV = "CSV2.csv"
separador = ";"

especie = "Species"
nombreIMG = "FileName"

rutaCopiarAnimales = "C:\TFG\CODIGOS\CODIGOS_VC\BBDD_2_PREPROC\\"

# Obtenemos un diccionario {nombreArchivo : especie} a partir del fichero CSV

# Parámetros
diccionario = {}

CSV = open(nombreCSV, encoding="utf8") 
reader = CSV.readlines()
primeraLinea = True

indiceEspecie = -1
indiceNombreIMG = -1

numLineas = 0

nombresRegistrosDicc = []

especies = set()


# Leemos el archivo CSV linea a linea
for linea in reader:

    # Eliminar salto de linea
    linea = linea.rstrip()

    # Para la primera línea, guardamos los índices de cada columna que nos interesa
    if primeraLinea:
        nombres = linea.split(';')
        indiceEspecie = nombres.index(especie)
        indiceNombreIMG = nombres.index(nombreIMG)
        primeraLinea = False

    # Para el resto de líneas guardamos los datos
    else:
        numLineas += 1
        datosFoto = linea.split(';')

        key = datosFoto[indiceNombreIMG]
        value = [datosFoto[indiceEspecie]]

        # Añadir elementos al diccionario
        diccionario[key] = value[0]

        nombresRegistrosDicc.append(key)
        especies.add(value[0])

        

# Mostramos los resultados
print("Lineas leidas: " + str(numLineas))
print("Longitud del diccionario: " + str(len(diccionario)))

print(diccionario)



#Como no nos interesan "especies" como NOID, vehicle, etc... vamos a no contarlas:
listaNoAnimales = ["human", "NOID", "vehicle","NA", "","empty"]  # Identificadores que aparecen en el archivo CSV


# Obtener más estadísticas
# Separar imágenes con animales de imágenes vacías


# Variables
noAparecenTotal = 0
noAparecen = 0
descarte = 0
total = 0
animals = 0
imagenesEncontradas = 0
imagenesEncontradasTotal = 0
listaImagenes = []


# Recorremos el directorio de imágenes
for root, dirs, files in os.walk(ruta, topdown=False):
   for name in files:


      imagenesEncontradas += 1
      imagenesEncontradasTotal += 1
      listaImagenes.append(name)
      
      # Si están en el diccionario significa que aparecen en el fichero CSV
      if diccionario.get(name):

         valor = diccionario.get(name)

         #print(valor)

         # Comprobamos si no están en la lista de descartes y aparezcan animales (suponiendo que si aparece un vehículo y/o persona no aparece un animal)
         if not (valor in listaNoAnimales):
            #print("Entra: "+valor)
            rutaIMG = os.path.join(root, name)
            
            # Abrimos con OpenCV para hacer la transformación
            img = cv.imread(rutaIMG)

            # Abrimos con Image para mantener los metadatos
            imWithEXIF = Image.open(rutaIMG)

            # Abrimos la imagen transformada con PIL
            pil1 = Image.fromarray(img)

            animals += 1

            # Copiar a Animales
            if 'exif' in imWithEXIF.info:
               exif = imWithEXIF.info['exif']
               pil1.save(rutaCopiarAnimales + name, format='JPEG', exif=exif)
            else:
               pil1.save(rutaCopiarAnimales + name, format='JPEG', exif=exif)

         # Si están en la lista de descartes, no guardamos la nueva imagen
         else:
            descarte += 1


      # Si no aparecen en el diccionario, no están en el fichero CSV. Descartamos  
      else:
         noAparecen += 1
         noAparecenTotal += 1
            

   # Mostramos estadísticas por directorio
   print("\n")
   print(os.path.basename(root))
   print("Número de imagenes encontradas: ", imagenesEncontradas)
   print("Número de imágenes que no aparecen en el archivo csv ", noAparecen)
   print("Número de imagenes que contienen animales: ", animals)

   # Reiniciamos variables
   total = 0
   animals = 0
   imagenesEncontradas = 0

# Mostramos estadísticas globales
print("Número total de imágenes: ", imagenesEncontradasTotal)
print("Número de imágenes descartadas: ", descarte)