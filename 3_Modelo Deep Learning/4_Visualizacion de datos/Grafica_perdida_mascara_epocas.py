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

def crear_lista_valores_perdida_mascara_IoU(ruta_archivo):

    fin = []
    fin_2 = []
    with open(ruta_archivo,"r") as archivo:
        contenido = archivo.read()
        lineas_contenido = contenido.split("  ")
        resultado = [i.split("\n") for i in lineas_contenido if "loss_mask_iou:" in i]
        for linea_1 in resultado:
            for linea_2 in linea_1:
                if "loss_mask_iou:" in linea_2:
                    valor_perdida = linea_2.split("loss_mask_iou: ")
                    fin_2.append(float(valor_perdida[1]))
                    if len(fin_2) >= PARTICIONES_ENTRENAMIENTO:
                        fin.append(fin_2)
                        fin_2 = []
    
    return fin


def media_por_epoca_perdidas_mascara(lista_todas_perdidas_mascara):
    lista_medias = []
    minimo = 10000
    for epoca in lista_todas_perdidas_mascara:
        media = 0.0
        for valor_perdida in epoca:
            media += valor_perdida
        media = media / len(epoca)
        if minimo > media:
            minimo = media
            print(minimo," en época ",len(lista_medias)) #Para ver la evolución de mínimos alcanzados en la función de pérdida.
        lista_medias.append(media)
    
    return lista_medias

def representar_medias_ocupacion(lista_medias,numero_de_funcion):
    
    nombres_barras = [i+1 for i in range(len(lista_medias))]

    plt.plot(nombres_barras, lista_medias)

    plt.xlabel('ÉPOCAS')
    plt.ylabel('Pérdida de la máscara IoU')
    if numero_de_funcion == 1:
        plt.title('Función de pérdida IoU - MaskRCNN - Contornos - Con entreno previo')
    elif numero_de_funcion == 2:
        plt.title('Función de pérdida - MaskRCNN - Contornos - Sin entreno previo')
    elif numero_de_funcion == 3:
        plt.title('Función de pérdida - MaskRCNN - Tolerancia - Con entreno previo')
    elif numero_de_funcion == 4:
        plt.title('Función de pérdida - MaskRCNN - Tolerancia - Sin entreno previo')
    elif numero_de_funcion == 5:
        plt.title('Función de pérdida IoU - MaskScoringRCNN - Contornos - Con entreno previo')
    elif numero_de_funcion == 6:
        plt.title('Función de pérdida IoU - MaskScoringRCNN - Contornos - Sin entreno previo')
    elif numero_de_funcion == 7:
        plt.title('Función de pérdida IoU - MaskScoringRCNN - Tolerancia - Con entreno previo')
    elif numero_de_funcion == 8:
        plt.title('Función de pérdida IoU - MaskScoringRCNN - Tolerancia - Sin entreno previo')

    # Fijar los límites del eje Y entre 0 y 1
    plt.ylim(0, 1)
    plt.xlim(0,300)

    plt.show()


#Mask R-CNN
ruta_documento_1 = r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Visualizacion de datos\maskrcnn_contornos_preentreno.log" #MaskRCNN, Contornos, con prentreno.
ruta_documento_2 = r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Visualizacion de datos\maskrcnn_contornos_sinpreentreno.log" #MaskRCNN, Contornos, sin prentreno.
ruta_documento_3 = r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Visualizacion de datos\maskrcnn_tol10_preentreno.log" #MaskRCNN, Tolerancia 10, con prentreno. 
ruta_documento_4 = r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Visualizacion de datos\maskrcnn_tol10_sinpreentreno.log" #MaskRCNN, Tolerancia 10, sin prentreno.

#Mask Scoring R-CNN
ruta_documento_5 = r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Visualizacion de datos\msrcnn_contornos_preentreno.log" #ScoreMaskRCNN, Contornos, con prentreno.
ruta_documento_6 = r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Visualizacion de datos\msrcnn_contornos_sinpreentreno.log" #ScoreMaskRCNN, Contornos, sin prentreno.
ruta_documento_7 = r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Visualizacion de datos\msrcnn_tol10_preentreno.log" #ScoreMaskRCNN, Tolerancia 10, con prentreno.
ruta_documento_8 = r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Visualizacion de datos\msrcnn_tol10_sinpreentreno.log" #ScoreMaskRCNN, Tolerancia 10, sin prentreno.


#Loss Mask
lista_perdidas_mascara_1 = crear_lista_valores_perdida_mascara(ruta_documento_1)
media_perdidas_mascara_1 = media_por_epoca_perdidas_mascara(lista_perdidas_mascara_1)
representar_medias_ocupacion(media_perdidas_mascara_1,1)


#Loss Mask
lista_perdidas_mascara_2 = crear_lista_valores_perdida_mascara(ruta_documento_2)
media_perdidas_mascara_2 = media_por_epoca_perdidas_mascara(lista_perdidas_mascara_2)
representar_medias_ocupacion(media_perdidas_mascara_2,2)


#Loss Mask
lista_perdidas_mascara_3 = crear_lista_valores_perdida_mascara(ruta_documento_3)
media_perdidas_mascara_3 = media_por_epoca_perdidas_mascara(lista_perdidas_mascara_3)
representar_medias_ocupacion(media_perdidas_mascara_3,3)

#Loss Mask
lista_perdidas_mascara_4 = crear_lista_valores_perdida_mascara(ruta_documento_4)
media_perdidas_mascara_4 = media_por_epoca_perdidas_mascara(lista_perdidas_mascara_4)
representar_medias_ocupacion(media_perdidas_mascara_4,4)


#Loss Mask
lista_perdidas_mascara_5 = crear_lista_valores_perdida_mascara(ruta_documento_5)
media_perdidas_mascara_5 = media_por_epoca_perdidas_mascara(lista_perdidas_mascara_5)
representar_medias_ocupacion(media_perdidas_mascara_5,5)
#Loss Mask IoU
lista_perdidas_mascara_IoU_5 = crear_lista_valores_perdida_mascara_IoU(ruta_documento_5)
media_perdidas_mascara_IoU_5 = media_por_epoca_perdidas_mascara(lista_perdidas_mascara_IoU_5)
representar_medias_ocupacion(media_perdidas_mascara_IoU_5,5)


#Loss Mask
lista_perdidas_mascara_6 = crear_lista_valores_perdida_mascara(ruta_documento_6)
media_perdidas_mascara_6 = media_por_epoca_perdidas_mascara(lista_perdidas_mascara_6)
representar_medias_ocupacion(media_perdidas_mascara_6,6)
#Loss Mask IoU
lista_perdidas_mascara_IoU_6 = crear_lista_valores_perdida_mascara_IoU(ruta_documento_6)
media_perdidas_mascara_IoU_6 = media_por_epoca_perdidas_mascara(lista_perdidas_mascara_IoU_6)
representar_medias_ocupacion(media_perdidas_mascara_IoU_6,6)


#Loss Mask
lista_perdidas_mascara_7 = crear_lista_valores_perdida_mascara(ruta_documento_7)
media_perdidas_mascara_7 = media_por_epoca_perdidas_mascara(lista_perdidas_mascara_7)
representar_medias_ocupacion(media_perdidas_mascara_7,7)
#Loss Mask IoU
lista_perdidas_mascara_IoU_7 = crear_lista_valores_perdida_mascara_IoU(ruta_documento_7)
media_perdidas_mascara_IoU_7 = media_por_epoca_perdidas_mascara(lista_perdidas_mascara_IoU_7)
representar_medias_ocupacion(media_perdidas_mascara_IoU_7,7)

#Loss Mask
lista_perdidas_mascara_8 = crear_lista_valores_perdida_mascara(ruta_documento_8)
media_perdidas_mascara_8 = media_por_epoca_perdidas_mascara(lista_perdidas_mascara_8)
representar_medias_ocupacion(media_perdidas_mascara_8,8)
#Loss Mask IoU
lista_perdidas_mascara_IoU_8 = crear_lista_valores_perdida_mascara_IoU(ruta_documento_8)
media_perdidas_mascara_IoU_8 = media_por_epoca_perdidas_mascara(lista_perdidas_mascara_IoU_8)
representar_medias_ocupacion(media_perdidas_mascara_IoU_8,8)