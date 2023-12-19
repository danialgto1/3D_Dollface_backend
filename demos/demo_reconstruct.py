# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

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

def cat_textures(savefolder, folder_name, opdict):
    net, tfms = load_model()

    front = util.tensor2image(opdict['uv_texture_gt'][0])
    left = cv2.imread(f'{savefolder}/{folder_name}/left.png')
    right = cv2.imread(f'{savefolder}/{folder_name}/right.png')
    black = cv2.imread('data/head.png', cv2.IMREAD_GRAYSCALE)
    dilatedBlack = cv2.dilate(black, np.ones((5, 5), np.uint8), iterations=2)
    bluredBlack = cv2.GaussianBlur(dilatedBlack, (9, 9), 10)
    meanTex = cv2.imread('data/mean_texture.jpg')
    meanTex = cv2.resize(meanTex, (256, 256))
    mask = cv2.imread('data/uv_face_eye_mask.png', cv2.IMREAD_GRAYSCALE)
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

def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)
    folder_name = testdata.folder_name
    assert len(testdata) == 3

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)
    os.makedirs(os.path.join(savefolder, folder_name), exist_ok=True)

    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict) #tensor
            if i < 2:
                texture = util.tensor2image(opdict['uv_texture_gt'][0])
                cv2.imwrite(os.path.join(savefolder, folder_name, name + '.png'), texture)
                continue
            
            else:
                opdict = cat_textures(savefolder, folder_name, opdict)

            if args.render_orig:
                tform = testdata[i]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to(device)
                original_image = testdata[i]['original_image'][None, ...].to(device)
                _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    
                orig_visdict['inputs'] = original_image            

        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            os.makedirs(os.path.join(savefolder, folder_name), exist_ok=True)
        # -- save results
        if args.saveDepth:
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, folder_name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if args.saveKpt:
            np.savetxt(os.path.join(savefolder, folder_name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, folder_name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, folder_name, name + '.obj'), opdict)
        if args.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, folder_name, name + '.mat'), opdict)
        if args.saveVis:
            cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
            if args.render_orig:
                cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                if vis_name not in visdict.keys():
                    continue
                image = util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(os.path.join(savefolder, folder_name, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
                if args.render_orig:
                    image = util.tensor2image(orig_visdict[vis_name][0])
                    cv2.imwrite(os.path.join(savefolder, folder_name, 'orig_' + name + '_' + vis_name +'.jpg'), util.tensor2image(orig_visdict[vis_name][0]))
    print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())