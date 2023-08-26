import mmcv
from mmdet.apis import init_detector, inference_detector
from mmengine import Config
import numpy as np
import os

def load_model(config_path, checkpoint_file):
    cfg = Config.fromfile(config_path)
    cfg.model.roi_head.bbox_head.num_classes = 1
    cfg.model.roi_head.mask_head.num_classes = 1
    model = init_detector(cfg, checkpoint_file, device='cpu')
    return model

def process_image(input_image_path, model, output_folder):
    image_name = os.path.basename(input_image_path)
    output_image_path = os.path.join(output_folder, image_name)

    if os.path.exists(output_image_path):
        print(f"Imagen ya procesada, omitiendo: {output_image_path}")
        return
    
    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"No es una imagen: {output_image_path}")
        return
    
    img = mmcv.imread(input_image_path, channel_order='rgb')

    result = inference_detector(model, img)

    try:
        mask_pred = result.pred_instances.masks[0].numpy()
        background = np.zeros_like(img)
        color = np.array([249, 249, 249])
        output_image = np.where(mask_pred[:, :, np.newaxis], color, background)
        output_path = os.path.join(output_folder, image_name)
        mmcv.imwrite(output_image, output_path)
        print("Imagen segmentada guardada en:", output_path)
    except:
        pass


if __name__ == "__main__":
    
    input_image_folder = r"C:\Users\Lenovo\Downloads\Dataset_pruebas\Images\test"
    output_folder = r"C:\Users\Lenovo\Downloads\resultados_segmentacion\version1_300epochs"
    os.makedirs(output_folder, exist_ok=True)
    checkpoint_file = r"C:\TFG\modelos_epochs\v1_epoch_300.pth"
    config_path = r"C:\TFG\modelos_epochs\version1_modelo_configuracion.py"
    
    model = load_model(config_path, checkpoint_file)

    for image_name in os.listdir(input_image_folder):
        input_image_path = os.path.join(input_image_folder, image_name)
        process_image(input_image_path, model, output_folder)
