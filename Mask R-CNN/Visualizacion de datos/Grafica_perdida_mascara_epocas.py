import matplotlib.pyplot as plt


NUMERO_EPOCAS = 300
PARTICIONES_ENTRENAMIENTO = 64

def crear_lista_valores_perdida_mascara(ruta_archivo):

    fin = []
    fin_2 = []
    with open(ruta_archivo,"r") as archivo:
        contenido = archivo.read()
        lineas_contenido = contenido.split("  ")
        resultado = [i.split("\n") for i in lineas_contenido if "loss_mask:" in i]
        for linea_1 in resultado:
            for linea_2 in linea_1:
                if "loss_mask:" in linea_2:
                    valor_perdida = linea_2.split("loss_mask: ")
                    fin_2.append(float(valor_perdida[1]))
                    if len(fin_2) >= PARTICIONES_ENTRENAMIENTO:
                        fin.append(fin_2)
                        fin_2 = []
    
    return fin


def media_por_epoca_perdidas_mascara(lista_todas_perdidas_mascara):
    lista_medias = []
    for epoca in lista_todas_perdidas_mascara:
        media = 0.0
        for valor_perdida in epoca:
            media += valor_perdida
        media = media / len(epoca)
        lista_medias.append(media)
    return lista_medias

def representar_medias_ocupacion(lista_medias):
    # Lista de nombres para las barras (puedes modificarla según tus necesidades)
    nombres_barras = [i+1 for i in range(len(lista_medias))]

    # Crear el gráfico de barras
    plt.bar(nombres_barras, lista_medias)

    # Agregar etiquetas y título al gráfico
    plt.xlabel('ÉPOCAS')
    plt.ylabel('Función de pérdida de la máscara')
    plt.title('Gráfico de pérdidas de la máscara por épocas')

    # Mostrar el gráfico
    plt.show()



ruta_documento = r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Visualizacion de datos\log.log"
lista_perdidas_mascara = crear_lista_valores_perdida_mascara(ruta_documento)

print(len(lista_perdidas_mascara))
media_perdidas_mascara = media_por_epoca_perdidas_mascara(lista_perdidas_mascara)

print(len(media_perdidas_mascara))

representar_medias_ocupacion(media_perdidas_mascara)

