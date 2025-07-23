# DISCLAIMER:
# 
# This code is part of the research paper titled "UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping."
# Use it freely, modify it, and build upon it in accordance with the terms of the CC BY-NC 4.0 License. 
# If you use or reference this work, please cite our paper.
# 
# For more information, please refer to the paper: https://arxiv.org/abs/2502.01846
# For updates, please visit the project website: https://aashishrai3799.github.io/uvgs
# 
# Copyright (C) 2025 Aashish Rai.
# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# https://creativecommons.org/licenses/by-nc/4.0
#


import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from uvgs_scripts.helper_functions import *


def psnr(true_img, pred_img, data_range=1.0):
    """
    Calculate the PSNR between two images.

    Args:
    true_img (torch.Tensor): The ground truth image tensor.
    pred_img (torch.Tensor): The predicted or reconstructed image tensor.
    data_range (float): The data range of the pixel values (default is 1.0, typical for normalized images).

    Returns:
    float: The PSNR value in decibels (dB).
    """
    # Ensure the input tensors are of the same shape
    if true_img.shape != pred_img.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    mse = torch.nn.functional.mse_loss(pred_img, true_img, reduction='mean')
    if mse == 0:
        # Avoid log(0) error by returning infinity
        return float('inf')
    
    # Calculate PSNR
    psnr_value = 20 * torch.log10(data_range / torch.sqrt(mse))
    
    return psnr_value.item()

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + std * eps

class gs_dataset(Dataset):
    def __init__(self, gs_path):
        
        self.gs_path = gs_path
        
        LIST_OF_CARS = []
        with open("LIST_OF_CAR_COLORS.txt", "r") as f:
            for line in f:
                LIST_OF_CARS.append(line.strip())

        self.file_names = [d for d in LIST_OF_CARS
                           if os.path.exists(os.path.join(self.gs_path, d.split(':')[0], 'point_cloud/iteration_10000/point_cloud.ply'))
                           ]
        
        # self.file_names = os.listdir(self.gs_path)
        # self.file_names = LIST_OF_CARS
        random.shuffle(self.file_names)
        print('\n\n\nDATASET SIZE:', len(self.file_names), '\n\n\n')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        
        file_name = self.file_names[idx].split(':')[0]
        NUM_POINTS = 256
        M = NUM_POINTS*NUM_POINTS

        # Load feature and latent code using numpy
        gs_ply_path = os.path.join(self.gs_path, file_name, 'point_cloud/iteration_10000/point_cloud.ply')
        input_features = gs_reader(gs_ply_path, NUM_POINTS) # dict of 3dgs attributes
        target = torch.cat((input_features['position_offset'],input_features['scale'],input_features['rotation'],input_features['opacity'][...,None],input_features['color']), axis=1)
        
        return target, file_name
    



class gs_dataset_all(Dataset):
    def __init__(self, gs_path, return_shs=False):
        
        self.gs_path = gs_path
        self.return_shs = return_shs
        
        self.file_names = [ f for f in os.listdir(self.gs_path) if os.path.exists(os.path.join(self.gs_path, f,"point_cloud/iteration_45000/point_cloud.ply") ) ]
        
        # self.file_names = os.listdir(self.gs_path)
        random.shuffle(self.file_names)
        print('\n\n\nDATASET SIZE:', len(self.file_names), '\n\n\n')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        
        file_name = self.file_names[idx]
        file_name = file_name + "/point_cloud/iteration_15000/point_cloud.ply"

        gs_ply_path = os.path.join(self.gs_path, file_name)
        input_features = gs_reader(gs_ply_path, self.return_shs) # dict of 3dgs attributes
        
        if self.return_shs:
            target = torch.cat((input_features['position_offset'],
                                input_features['scale'],
                                input_features['rotation'],
                                input_features['opacity'][...,None],
                                input_features['color'],
                                input_features['f_rest']), 
                               axis=1)
        
        else:
            target = torch.cat((input_features['position_offset'],
                                input_features['scale'],
                                input_features['rotation'],
                                input_features['opacity'][...,None],
                                input_features['color']), 
                               axis=1)
            
        
        return target, self.file_names[idx]


def load_gs_from_file(root_dir, iter='10000', return_shs=True):
    
    gs_ply_path = os.path.join(root_dir, f"point_cloud/iteration_{iter}/point_cloud.ply")
    input_features = gs_reader(gs_ply_path, return_shs)
    
    if return_shs:
        target = torch.cat((input_features['position_offset'],
                            input_features['scale'],
                            input_features['rotation'],
                            input_features['opacity'][...,None],
                            input_features['color'],
                            input_features['f_rest']), 
                            axis=1)
    
    else:
        target = torch.cat((input_features['position_offset'],
                            input_features['scale'],
                            input_features['rotation'],
                            input_features['opacity'][...,None],
                            input_features['color']), 
                            axis=1)
        
    return target


    
    
def uv_gs_feat_norm(uv_gs):    # Scaling Function
    
    uv_gs[:,:,:3] = uv_gs[:,:,:3]  / 2
    # uv_gs[:,:,3:6] = torch.exp(uv_gs[:,:,3:6])*2-1
    # uv_gs[:,:,3:6] = (1-torch.exp(uv_gs[:,:,3:6]))*2-1
    uv_gs[:,:,3:6] = ( uv_gs[:,:,3:6].clip(-10,0) + 5) / 5
    uv_gs[:,:,6:10] = uv_gs[:,:,6:10] / 3
    uv_gs[:,:,10:11] = uv_gs[:,:,10:11] / 20
    uv_gs[:,:,11:14] = (uv_gs[:,:,11:14] / 4.2).clip(-1,1)
    
    return uv_gs

def uv_gs_feat_un_norm(uv_gs):   # Scaling Function
    
    uv_gs[:,:,:3] = uv_gs[:,:,:3]  * 2
    # uv_gs[:,:,3:6] = torch.log( ( uv_gs[:,:,3:6] + 1.000001 ) / 2 )
    # uv_gs[:,:,3:6] = torch.log( ( 1.000001 - uv_gs[:,:,3:6] ) / 2 )
    uv_gs[:,:,3:6] = ( uv_gs[:,:,3:6] * 5 ) - 5 
    uv_gs[:,:,6:10] = uv_gs[:,:,6:10] * 3
    uv_gs[:,:,10:11] = uv_gs[:,:,10:11] * 20
    uv_gs[:,:,11:] = (uv_gs[:,:,11:] * 4.2)
    
    return uv_gs
    
    
class gs_uv_dataset(Dataset):

    def __init__(self, gs_path, returns_filename=False):
        
        self.returns_filename = returns_filename
        self.gs_path = gs_path
        K=1
        
        self.file_names = [f for f in os.listdir(self.gs_path) if f'_{K}.npy' in f]
        random.shuffle(self.file_names)
        
        print('\n\n\nDATASET SIZE:', len(self.file_names), '\n\n\n')


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        
        file_name = self.file_names[idx]
        uv_gs_features = np.array(np.load( os.path.join( self.gs_path, file_name ) ), dtype='float32')
        
        # Scale GS Attributes
        uv_gs_feats_norm = uv_gs_feat_norm(torch.from_numpy(uv_gs_features))
        
        if self.returns_filename:
            return uv_gs_feats_norm, file_name
            
        else:
            return uv_gs_feats_norm 



class gs_uv_dataset_multilayer(Dataset):
    def __init__(self, gs_path, K=4, mapping_type='S', returns_filename=False):
        
        # list_of_files = ['baf1c43f224f47f19fb4cb6d0033bfd1_gs.ply', '323a58196eaa43f3aa1bfc4716188c15_gs.ply']
        self.returns_filename = returns_filename
        self.gs_path = gs_path
        self.mapping_type = mapping_type
        self.file_names = [f for f in os.listdir(self.gs_path) if f'_{K}_{mapping_type}.npy' in f]
        
        random.shuffle(self.file_names)
        
        print('\n\n\nDATASET SIZE:', len(self.file_names), '\n\n\n')


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        
        file_name = self.file_names[idx]
        uv_gs_features = np.array(np.load( os.path.join( self.gs_path, file_name ) ), dtype='float32')
        
        # Scale GS Attributes
        # uv_gs_feats_norm = uv_gs_feat_norm(torch.from_numpy(uv_gs_features))
        uv_gs_feats_norm = torch.from_numpy(uv_gs_features)
        
        if self.returns_filename:
            return uv_gs_feats_norm, file_name
            
        else:
            return uv_gs_feats_norm 
    



class uvgs_img_dataset(Dataset):
    def __init__(self, gs_path, gs_img_path, returns_filename=False):
        
        self.returns_filename = returns_filename
        self.gs_path = gs_path
        self.gs_img_path = gs_img_path
        self.file_names = os.listdir(self.gs_path)
        
        random.shuffle(self.file_names)
        
        print('\n\n\nDATASET SIZE:', len(self.file_names), '\n\n\n')


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        
        file_name = self.file_names[idx]
        uv_gs_features = np.array(np.load( os.path.join( self.gs_path, file_name ) ), dtype='float32')
        
        # Scale GS Attributes
        uv_gs_feats_norm = uv_gs_feat_norm(torch.from_numpy(uv_gs_features))
        
        # Load Image
        img_path = os.path.join( self.gs_img_path, file_name.split('.')[0], "render/055.png" )
        img = ( np.array(Image.open(img_path)) / 127.5 ) - 1
        
        if self.returns_filename:
            return uv_gs_feats_norm, img[:,:,:3], file_name
            
        else:
            return uv_gs_feats_norm, img[:,:,:3] 
    
    
    
    