import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy.io import savemat
import shutil
from home.apps import cfg
from django.core.files.base import ContentFile


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.tensor_cropper import transform_points



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
        print('done')
    print(f'-- please check the results in {config.savefolder}')

def create_3d(instance):
    path = [instance.left.path , instance.right.path , instance.front.path]
    cfg.savefolder = os.path.join(os.path.dirname(instance.left.path) , 'tmp')
    main(path , cfg)
    obj_path = os.path.join(cfg.savefolder,'images','front.obj')
    mtl_path = os.path.join(cfg.savefolder,'images','front.mtl')
    texture_path = os.path.join(cfg.savefolder,'images','front.png')
    print(obj_path)
    if os.path.exists(obj_path):
        with open(obj_path, 'rb') as file:
                    buf = file.read()
                    content_file = ContentFile(buf, 'front.obj')
        instance.obj = content_file
    
        with open(texture_path, 'rb') as file:
                    buf = file.read()
                    content_file = ContentFile(buf, 'front.png')
        instance.texture = content_file

        with open(mtl_path, 'rb') as file:
                    buf = file.read()
                    content_file = ContentFile(buf, 'front.mtl')

        instance.mtl = content_file
        instance.save()
        shutil.rmtree(cfg.savefolder)
    else: raise ValueError()
    