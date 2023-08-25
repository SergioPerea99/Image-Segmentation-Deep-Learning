import mmcv
from mmdet.apis import init_detector, inference_detector
from mmengine import Config
import numpy as np
import os

# Inicialización de variables
input_image_folder = r"C:\Users\Lenovo\Downloads\Dataset_pruebas\Images\test"
output_folder = r"C:\Users\Lenovo\Downloads\resultados_segmentacion\version1_300epochs"
os.makedirs(output_folder, exist_ok=True)
checkpoint_file = r"C:\TFG\modelos_epochs\v1_epoch_300.pth"
cfg = Config.fromfile(r"C:\TFG\modelos_epochs\version1_modelo_configuracion.py")
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1
model = init_detector(cfg, checkpoint_file, device='cpu') # Cargar el modelo una vez 


# Ejecución de todas las imágenes
for image_name in os.listdir(input_image_folder):

    output_image_path = os.path.join(output_folder, image_name)
    
    if os.path.exists(output_image_path):
        print(f"Imagen ya procesada, omitiendo: {output_image_path}")
        continue
    
    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"No es una imagen: {output_image_path}")
        continue
    
    img = mmcv.imread(os.path.join(input_image_folder, image_name), channel_order='rgb')

    # Realizar la inferencia en la imagen
    result = inference_detector(model, img)

    try:
        # Acceder a la máscara de segmentación de la predicción
        mask_pred = result.pred_instances.masks[0].numpy()

        # Crear una imagen segmentada aplicando la máscara a una imagen de fondo negra
        background = np.zeros_like(img)  
        color = np.array([249,249,249])

        output_image = np.where(mask_pred[:, :, np.newaxis], color, background)

        # Guardar la imagen segmentada en la carpeta de salida
        output_path = os.path.join(output_folder,image_name)
        mmcv.imwrite(output_image, output_path)

        print("Imagen segmentada guardada en:", output_path)
    except:
        continue
