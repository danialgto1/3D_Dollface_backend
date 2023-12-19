from django.apps import AppConfig
import os
import sys
import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

        
def load_model(seg_path ='data/Face_DeepLabV3+_TimmRegnety002.pth'):
    tfms = A.Compose([
        A.Resize(480, 480),
        ToTensorV2()
    ])

    net = smp.DeepLabV3Plus(
        encoder_name='timm-regnety_002',
        encoder_weights=None,
        classes=4,
        in_channels=3,
        activation='softmax2d',
    )
    net.load_state_dict(torch.load(seg_path, map_location=torch.device('cpu'))['state_dict'])
    net.eval()

    return net, tfms

class config:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iscrop = True
    detector = 'fan'
    sample_step = 10
    useTex = True
    extractTex = True
    rasterizer_type = 'standard'
    render_orig = True
    saveDepth = True
    saveObj = True
    saveVis = True
    saveKpt = False
    saveMat = False
    saveImages = False

    # run DECA
    deca_cfg.model.use_tex = useTex
    deca_cfg.rasterizer_type = rasterizer_type
    deca_cfg.model.extract_tex = extractTex
    deca = DECA(config = deca_cfg, device=device)
    net, tfms = load_model()

class HomeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'home'

    def ready(self) -> None:
        global cfg
        cfg = config()
        return super().ready()