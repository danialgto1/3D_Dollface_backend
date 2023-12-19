import os
import sys
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from scipy.io import savemat

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

def load_model(seg_path ='../data/Face_DeepLabV3+_TimmRegnety002.pth'):
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

def face_segment(img, net, tfms):
    img = tfms(image=img)['image'].unsqueeze(0)/255
    with torch.no_grad():
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        parsing1 = (parsing == 1).astype(np.uint8)*255
        parsing2 = (parsing == 2).astype(np.uint8)*255
        parsing = cv2.bitwise_or(parsing1, parsing2)
        resized_img = cv2.resize(parsing, (240, 240), interpolation=cv2.INTER_NEAREST)
        final_seg = np.zeros((256, 256), dtype=np.uint8)
        final_seg[8:-8, 8:-8] = resized_img
    return final_seg

def cat_textures(savefolder, folder_name, net, tfms, opdict):
    front = util.tensor2image(opdict['uv_texture_gt'][0])
    left = cv2.imread(f'{savefolder}/{folder_name}/left.png')
    right = cv2.imread(f'{savefolder}/{folder_name}/right.png')
    black = cv2.imread('../data/head.png', cv2.IMREAD_GRAYSCALE)
    dilatedBlack = cv2.dilate(black, np.ones((5, 5), np.uint8), iterations=2)
    bluredBlack = cv2.GaussianBlur(dilatedBlack, (9, 9), 10)
    meanTex = cv2.imread('../data/mean_texture.jpg')
    meanTex = cv2.resize(meanTex, (256, 256))
    mask = cv2.imread('../data/uv_face_eye_mask.png', cv2.IMREAD_GRAYSCALE)
    mask = (mask > 100).astype(np.float32).astype(np.uint8)*255
    mask[:47] = 0

    frontRGB = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
    segmented_face = face_segment(frontRGB, net, tfms)
    sideMask = cv2.bitwise_xor(segmented_face, mask)
    bluredSideMask = cv2.GaussianBlur(sideMask, (9, 9), 10)
    W = sideMask.shape[1]

    sidePic = np.zeros((256, 256, 3), dtype=np.uint8)
    sidePic[:, :W//2] = right[:, :W//2]
    sidePic[:, W//2:] = left[:, W//2:]

    a_B, a_G, a_R = cv2.split(sidePic)
    b_B, b_G, b_R = cv2.split(front)

    b = (a_B * (bluredSideMask/255.0)) + (b_B * (1.0 - (bluredSideMask/255.0)))
    g = (a_G * (bluredSideMask/255.0)) + (b_G * (1.0 - (bluredSideMask/255.0)))
    r = (a_R * (bluredSideMask/255.0)) + (b_R * (1.0 - (bluredSideMask/255.0)))

    a_B, a_G, a_R = cv2.split(meanTex)
    b = (a_B * (bluredBlack/255.0)) + (b * (1.0 - (bluredBlack/255.0)))
    g = (a_G * (bluredBlack/255.0)) + (g * (1.0 - (bluredBlack/255.0)))
    r = (a_R * (bluredBlack/255.0)) + (r * (1.0 - (bluredBlack/255.0)))

    finalImg = cv2.merge((b, g, r)).astype(np.uint8)

    opdict['cat_tex'] = finalImg

    return opdict

class config:
    savefolder = 'results'
    
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

def main(inputpath, config):
    # load test images 
    testdata = datasets.TestData(inputpath, iscrop=config.iscrop, face_detector=config.detector, sample_step=config.sample_step)
    folder_name = testdata.folder_name
    assert len(testdata) == 3
    os.makedirs(os.path.join(config.savefolder, folder_name), exist_ok=True)


    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(config.device)[None,...]
        with torch.no_grad():
            codedict = config.deca.encode(images)
            opdict, visdict = config.deca.decode(codedict) #tensor
            if i < 2:
                texture = util.tensor2image(opdict['uv_texture_gt'][0])
                cv2.imwrite(os.path.join(config.savefolder, folder_name, name + '.png'), texture)
                continue
            
            else:
                opdict = cat_textures(config.savefolder, folder_name, config.net, config.tfms, opdict)

            if config.render_orig:
                tform = testdata[i]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to(config.device)
                original_image = testdata[i]['original_image'][None, ...].to(config.device)
                _, orig_visdict = config.deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    
                orig_visdict['inputs'] = original_image            

        if config.saveDepth or config.saveKpt or config.saveObj or config.saveMat or config.saveImages:
            os.makedirs(os.path.join(config.savefolder, folder_name), exist_ok=True)
        # -- save results
        if config.saveDepth:
            depth_image = config.deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(config.savefolder, folder_name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if config.saveKpt:
            np.savetxt(os.path.join(config.savefolder, folder_name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(config.savefolder, folder_name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        if config.saveObj:
            config.deca.save_obj(os.path.join(config.savefolder, folder_name, name + '.obj'), opdict)
        if config.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(config.savefolder, folder_name, name + '.mat'), opdict)
        if config.saveVis:
            cv2.imwrite(os.path.join(config.savefolder, name + '_vis.jpg'), config.deca.visualize(visdict))
            if config.render_orig:
                cv2.imwrite(os.path.join(config.savefolder, name + '_vis_original_size.jpg'), config.deca.visualize(orig_visdict))
        if config.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                if vis_name not in visdict.keys():
                    continue
                image = util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(os.path.join(config.savefolder, folder_name, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
                if config.render_orig:
                    image = util.tensor2image(orig_visdict[vis_name][0])
                    cv2.imwrite(os.path.join(config.savefolder, folder_name, 'orig_' + name + '_' + vis_name +'.jpg'), util.tensor2image(orig_visdict[vis_name][0]))
    print(f'-- please check the results in {config.savefolder}')

path= ['/mnt/newdisk/python/projects/december/3d-model/core/media/uploads/1/right.jpeg', '/mnt/newdisk/python/projects/december/3d-model/core/media/uploads/1/left.jpeg','/mnt/newdisk/python/projects/december/3d-model/core/media/uploads/1/front.jpeg']
# path = os.path.join('/mnt/newdisk/python/projects/december/3d-model/core/media/uploads/' , '1')
cfg1 = config()
cfg1.savefolder = os.path.dirname(path[0])
main(path,cfg1)
