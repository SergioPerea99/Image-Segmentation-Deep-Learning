import mmcv
import torch
import mmdet
import mmengine
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.runner import Runner


#Comprobacion previa de tener instalado lo necesario
print("torch version:",torch.__version__, "cuda:",torch.cuda.is_available())
print("mmdetection:",mmdet.__version__)
print("mmcv:",mmcv.__version__)
print("mmengine:",mmengine.__version__)


#Configuración previa 
cfg = Config.fromfile('./mmdetection/configs/ms_rcnn/ms-rcnn_r50-caffe_fpn_2x_coco.py')
cfg.metainfo = {
    'classes': ('animal', ),
    'palette': [
        (37, 177, 90),
    ]
}

cfg.optim_wrapper.optimizer.lr = 0.02 / 8
cfg.default_hooks.logger.interval = 10
set_random_seed(0, deterministic=False)


# Configuración de rutas y datasets
cfg.data_root = './Dataset/Images'
cfg.train_dataloader.dataset.ann_file = 'train/annotation_coco_contornos.json'
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = 'train/'
cfg.train_dataloader.dataset.metainfo = cfg.metainfo
cfg.val_dataloader.dataset.ann_file = 'val/annotation_coco_contornos.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = 'val/'
cfg.val_dataloader.dataset.metainfo = cfg.metainfo
cfg.test_dataloader = cfg.val_dataloader
cfg.work_dir = './tutorial_exps_score_300epochs_prentrenado_contornos'
cfg.train_cfg.val_interval = 3 #Cada 3 epochs, se evalua el conj. de val
cfg.default_hooks.checkpoint.interval = 50 #Configurar para cada cuanto se guarda el modelo.

# Configuración de métricas
cfg.val_evaluator.ann_file = cfg.data_root+'/'+'val/annotation_coco_contornos.json'
cfg.test_evaluator = cfg.val_evaluator

# Configuración del número de clases
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# Configuración de Transferencia de aprendizaje o no.
cfg.model.backbone.init_cfg = False
cfg.load_from = './mmdetection/checkpoints/ms_rcnn_r101_caffe_fpn_2x_coco_bbox_mAP-0.411__segm_mAP-0.381_20200506_011134-5f3cc74f.pth' #1ª opción de modelo pre-entrenado (git por defecto)

# Construimos el modelo
runner = Runner.from_cfg(cfg)

# Comenzamos el entrenamiento
runner.train()

