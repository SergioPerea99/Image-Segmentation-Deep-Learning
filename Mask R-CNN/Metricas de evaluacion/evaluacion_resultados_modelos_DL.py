import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS  # Contiene los visualizadores registrados en MMDetection
from mmengine import Config
from mmengine.runner import set_random_seed
import numpy as np

# Cargar la configuración del archivo
cfg = Config.fromfile(r"C:\TFG\3_MODELOS_DL\mmdetection\configs\ms_rcnn\ms-rcnn_r50-caffe_fpn_2x_coco.py")

# Modificar el número de clases de detección de cajas y en la de segmentación
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# Configurar el directorio de trabajo para guardar archivos y registros
cfg.work_dir = './tutorial_exps_score_300epochs_prentrenado_contornos'

# Importar el motor de ejecución y crear un runner a partir de la configuración
from mmengine.runner import Runner
runner = Runner.from_cfg(cfg)

# Cargar la imagen a segmentar
img = mmcv.imread(r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Metricas de evaluacion\Ca.02.003.2011.Babel(29-01)1_2_base.JPG", channel_order='rgb')

# Cargar el archivo de punto de control del modelo
checkpoint_file = r"C:\epoch_200.pth"
model = init_detector(cfg, checkpoint_file, device='cpu')

# Realizar la inferencia en la imagen
result = inference_detector(model, img)  # Obtenemos un DetDataSample que contiene la detección del animal
print(result)

# Acceder a la máscara de segmentación de la predicción
mask_pred = result.pred_instances.masks[0].numpy()
print(mask_pred)  # 'mask_pred' es ahora una matriz NumPy que representa la máscara de segmentación de la predicción

# Visualizar las áreas detectadas en la imagen original
import matplotlib.pyplot as plt

plt.imshow(img)
plt.imshow(mask_pred, alpha=0.5)  # Superponer la máscara en la imagen
plt.show()


# Crear una matriz con el valor de fondo (0, 0, 0) del mismo tamaño que la imagen original
background = np.zeros_like(img)

# Crear una matriz con el valor de RGB deseado (37, 177, 90) del mismo tamaño que la imagen original
color = np.ones_like(img) * np.array([37, 177, 90])

# Aplicar la máscara para cambiar los valores
output_image = np.where(mask_pred[:, :, np.newaxis], color, background)

# Guardar la imagen resultante en una ruta específica
output_path = r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Metricas de evaluacion\output_image.png"
mmcv.imwrite(output_image, output_path)

print("Imagen guardada en:", output_path)

# Dimensiones de la imagen resultante
print("Dimensiones de la imagen segmentada: ", output_image.shape)

# Mostrar la imagen de salida
plt.imshow(output_image)
plt.show()


label = mmcv.imread(r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Metricas de evaluacion\Ca.02.003.2011.Babel(29-01)1_2.JPG", channel_order='rgb')


# Crear una matriz con todos los píxeles en negro (fondo)
background = np.zeros_like(label)

# Crear una matriz con el valor de RGB deseado (37, 177, 90) del mismo tamaño que la imagen original
color = np.ones_like(label) * np.array([37, 177, 90])

# Crear una máscara booleana para identificar los píxeles segmentados (color igual a (37, 177, 90))
mask_segmented = np.all(label == color, axis=-1)

# Asignar el color deseado a los píxeles segmentados en la imagen de fondo negra
output_image_with_bg = np.where(mask_segmented[:, :, np.newaxis], color, background)

# Guardar la imagen resultante en una ruta específica
output_path = r"C:\TFG\GITHUB_MI_TFG\Image-Segmentation-Deep-Learning\Mask R-CNN\Metricas de evaluacion\output_image_reference.png"
mmcv.imwrite(output_image_with_bg, output_path)