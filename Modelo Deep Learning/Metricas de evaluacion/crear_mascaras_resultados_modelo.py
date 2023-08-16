import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmengine import Config
from mmengine.runner import set_random_seed
import numpy as np
import os

# Rutas necesarias
input_image_folder = r"C:\Users\Lenovo\Downloads\Dataset_pruebas\Images\test"
output_folder = r"C:\Users\Lenovo\Downloads\resultados_segmentacion"
os.makedirs(output_folder, exist_ok=True)

# Configuración del modelo
checkpoint_file = r"C:\epoch_200.pth"
cfg = Config.fromfile(r"C:\TFG\3_MODELOS_DL\mmdetection\configs\ms_rcnn\ms-rcnn_r50-caffe_fpn_2x_coco.py")
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
    
    # Cargar la imagen a segmentar
    img = mmcv.imread(os.path.join(input_image_folder, image_name), channel_order='rgb')

    # Realizar la inferencia en la imagen
    result = inference_detector(model, img)
    try:
        # Acceder a la máscara de segmentación de la predicción
        mask_pred = result.pred_instances.masks[0].numpy()

        # Crear una imagen segmentada aplicando la máscara a una imagen de fondo negra
        background = np.zeros_like(img)
        color = np.array([37, 177, 90])
        output_image = np.where(mask_pred[:, :, np.newaxis], color, background)

        # Guardar la imagen segmentada en la carpeta de salida
        output_path = os.path.join(output_folder,image_name)
        mmcv.imwrite(output_image, output_path)

        print("Imagen segmentada guardada en:", output_path)
    except:
        continue


