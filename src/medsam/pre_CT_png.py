#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% import packages
import numpy as np
import os
join = os.path.join 
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse

# set up the parser
parser = argparse.ArgumentParser(description='preprocess PNG images')
parser.add_argument('-i', '--img_path', type=str, default='./../../data/RaidiumImages/X_train', help='path to the PNG images')
parser.add_argument('-gt', '--gt_path', type=str, default='./../../data/RaidiumImages/Supp_train/segmentations', help='path to the ground truth',)
parser.add_argument('-o', '--npz_path', type=str, default='./../../data/RaidiumImages/NPZ', help='path to save the npz files')

parser.add_argument('--image_size', type=int, default=512, help='image size')
parser.add_argument('--modality', type=str, default='CT', help='modality')
parser.add_argument('--anatomy', type=str, default='Abd-Gallbladder', help='anatomy')
parser.add_argument('--img_extension', type=str, default='.png', help='image file extension')
parser.add_argument('--prefix', type=str, default='CT_Abd-Gallbladder_', help='prefix')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--checkpoint', type=str, default='work_dir/SAM/sam_vit_b_01ec64.pth', help='checkpoint')
parser.add_argument('--device', type=str, default='cpu', help='device')
# seed
parser.add_argument('--seed', type=int, default=2023, help='random seed')
args = parser.parse_args()

prefix = args.modality + '_' + args.anatomy
names = os.listdir(args.gt_path)
sorted_filenames = sorted(names, key=lambda x: int(x.split('.')[0]))[:200]

names = [name for name in sorted_filenames if not os.path.exists(join(args.npz_path, prefix + '_' + name.split('.png')[0]+'.npz'))]
names = [name for name in sorted_filenames if os.path.exists(join(args.img_path, name.split('.png')[0] + args.img_extension))]

# split names into training and testing
np.random.seed(args.seed)
np.random.shuffle(names)
train_names = sorted(names[:int(len(names)*0.8)])
test_names = sorted(names[int(len(names)*0.8):])

# def preprocessing function
def preprocess_png(gt_path, img_path, gt_name, image_name, image_size, sam_model, device='cuda:0'):
    gt_data = io.imread(join(gt_path, gt_name))
    gt_data = np.uint8(gt_data > 0)

    if np.sum(gt_data)>1000:
        imgs = []
        gts =  []
        img_embeddings = []
        assert np.max(gt_data)==1 and np.unique(gt_data).shape[0]==2, 'ground truth should be binary'
        image_data = io.imread(join(img_path, image_name))
        
        gt_resized = transform.resize(gt_data, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
        img_resized = transform.resize(image_data, (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
        assert len(img_resized.shape)==3 and img_resized.shape[2]==3, 'image should be 3 channels'
        assert img_resized.shape[0]==gt_resized.shape[0] and img_resized.shape[1]==gt_resized.shape[1], 'image and ground truth should have the same size'
        imgs.append(img_resized)
        assert np.sum(gt_resized)>100, 'ground truth should have more than 100 pixels'
        gts.append(gt_resized)
        
        if sam_model is not None:
            sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
            resize_img = sam_transform.apply_image(img_resized)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:])
            assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to the specified size'
            
            with torch.no_grad():
                embedding = sam_model.image_encoder(input_image)
                img_embeddings.append(embedding.cpu().numpy()[0])

    if sam_model is not None:
        return imgs, gts, img_embeddings
    else:
        return imgs, gts

#%% prepare the save path
save_path_tr = join(args.npz_path, prefix, 'train')
save_path_ts = join(args.npz_path, prefix, 'test')
os.makedirs(save_path_tr, exist_ok=True)
os.makedirs(save_path_ts, exist_ok=True)

#%% set up the model
sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)

for name in tqdm(train_names):
    image_name = name
    gt_name = name
    imgs, gts, img_embeddings = preprocess_png(args.gt_path, args.img_path, gt_name, image_name, args.image_size, sam_model, device=args.device)
    #%% save to npz file
    if len(imgs)>1:
        imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
        gts = np.stack(gts, axis=0) # (n, 256, 256)
        img_embeddings = np.stack(img_embeddings, axis=0) # (n, 1, 256, 64, 64)
        np.savez_compressed(join(save_path_tr, prefix + '_' + gt_name.split('.png')[0]+'.npz'), imgs=imgs, gts=gts, img_embeddings=img_embeddings)

for name in tqdm(test_names):
    image_name = name
    gt_name = name
    imgs, gts = preprocess_png(args.gt_path, args.img_path, gt_name, image_name, args.image_size, sam_model=None, device=args.device)
    #%% save to npz file
    if len(imgs)>1:
        imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
        gts = np.stack(gts, axis=0) # (n, 256, 256)
        np.savez_compressed(join(save_path_ts, prefix + '_' + gt_name.split('.png')[0]+'.npz'), imgs=imgs, gts=gts)

