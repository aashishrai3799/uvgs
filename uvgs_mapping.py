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
import cv2
import torch
import numpy as np
from natsort import natsorted

from uvgs_scripts.gs_dataloader import *
from uvgs_scripts.mapping_functions import *
from uvgs_scripts.helper_functions import prune_points_counter, spherical_unwrap_count_K_prune_mask, spherical_unwrap_valid_mask



def GetUVGSmap(
    gs_path,
    K=32,
    return_shs=True,
    uvgs_size=1024,
    mapping_type='S',
    device='cpu'
):
    """
    Generate UV mappings from a 3DGS scene and save generated UV maps as numpy files.

    Args:
        Ks (list or iterable):
            A list of integers specifying the 'K' values (number of mapping layers) to generate.
        device (str):
            The torch device to use, e.g. 'cpu' or 'cuda'.
        batch_size (int):
            Number of samples per batch in the DataLoader.
        return_shs (bool):
            Whether to load SH coefficients from the dataset.
        uvgs_size (int):
            Height and width of the UV map to generate (square shape).
        mapping_type (str):
            Type of mapping to perform. 'R' (equirectangular), 'S' (spherical),
            or 'RS' (both). If 'RS', results are concatenated.
        gs_path (str):
            Path to the 3DGS scene data (input).
        uv_gs_folder (str):
            Path to the output folder where .npy files will be saved.
    """
    

    # Load PLY file
    input_data = load_gs_from_file(gs_path, return_shs=False)


    # For each requested K, generate the UV mapping
    # Extract channels from input_data
    gs_points = input_data[:, :].detach().cpu().numpy()    # All Attributes
    points = input_data[:, :3].detach().cpu().numpy()      # XYZ
    opacity = input_data[:, 10:11].detach().cpu().numpy()  # Opacity

    # Arbitrary assignment from original code
    gs_points[0, :] = 0

    # Prepare for top-K unwrapping
    if mapping_type == 'S':
        
        # UV_mapping_S = fast_spherical_unwrap_topK_opacity(points, opacity, height=uvgs_size, width=uvgs_size, K=K)
        UV_mapping_S = fast_spherical_unwrap_fill_channels(points, height=uvgs_size, width=uvgs_size, K=K)
        
        UV_gs_points_topK_S = np.stack(
            [gs_points[UV_mapping_S[:, :, k]] for k in range(K)],
            axis=-1
        )
        UV_gs_points_topK = UV_gs_points_topK_S
        
        return UV_gs_points_topK
        



def GetUVGSmap_Fast(
    gs_path,
    K=32,
    return_shs=True,
    uvgs_size=1024,
    mapping_type='S',
    device='cpu'
):
    """
    Generate UV mappings from a 3DGS scene.

    Args:
        gs_path (str): Path to the 3DGS scene data (input).
        K (int): Number of mapping layers.
        return_shs (bool): (Unused in example) Whether to load SH coefficients.
        uvgs_size (int): UV map width and height.
        mapping_type (str): Type of mapping to perform. 'R' (equirectangular), 'S' (spherical),
                            or 'RS' (both). If 'RS', results are concatenated.
        device (str): Torch device, e.g. 'cpu' or 'cuda'.
    """
    
    # Load data
    input_data = load_gs_from_file(gs_path, iter='45000', return_shs=return_shs)

    # If you're using a GPU and want to keep data on GPU, you can leave it as a torch tensor.
    # Otherwise, move it to CPU and convert to numpy for the unwrapping:
    input_data = input_data.to(device)
    data_np = input_data.cpu().numpy()  # shape = (N, channels)

    # Extract the relevant attributes
    gs_points = data_np            # shape (N, all_channels)
    points    = data_np[:, :3]     # (x, y, z) for spherical unwrapping
    opacity   = data_np[:, 10:11]  # Opacity

    # Arbitrary assignment to simplify mapping
    gs_points[0, :] = 0

    # Only one case shown ('S', Spherical Projection), but adapt if you need others. Other mapping functions are defined in ./uvgs_scripts/mapping_functions.py
    if mapping_type == 'S':
        # Suppose this returns an (H, W, K) integer array of indices
        UV_mapping_S = fast_spherical_unwrap_topK_opacity(points, opacity, height=uvgs_size, width=uvgs_size, K=K)

        H, W, K_ = UV_mapping_S.shape
        # Flatten (H,W,K) => (H*W*K) for indexing:
        flat_idx = UV_mapping_S.reshape(-1)  # shape (H*W*K,)
        # gs_points[flat_idx] => shape (H*W*K, num_channels)
        mapped_points = gs_points[flat_idx]
        # Reshape back to (H, W, K, num_channels)
        mapped_points = mapped_points.reshape(H, W, K_, -1)
        # Original code expects (H, W, num_channels, K), so transpose:
        mapped_points = np.transpose(mapped_points, (0, 1, 3, 2))
        # shape = (H, W, channels, K)

        return mapped_points



if __name__ == "__main__":
    
    device = 'cpu'

    gs_path = "./fillted_3DGS_scene"       # path to fitted 3DGS scenes
    out_path = "./test_UV_Maps"            # path to output UVGS files

    os.makedirs(out_path, exist_ok=True)
    K = 8                                  # number of UVGS layers
    
    SAVE_UVGS = True
    SAVE_PLY = True

    gs_folders = os.listdir(gs_path)
    gs_folders = natsorted(gs_folders)
    
    # iterate over different fitted 3DGS scenes.
    for i, folder in enumerate( gs_folders ):
            
        print(f'Processing K: {K} || Index K: {folder} || {i}')
        
        root_gs_path = os.path.join(gs_path, folder)
        uvgs_map = GetUVGSmap_Fast(root_gs_path, K=K, return_shs=False, uvgs_size=1024, mapping_type='S', device=device)

        if SAVE_UVGS:
            np.save( os.path.join(out_path, folder+'.npy'), uvgs_map )
        
        if SAVE_PLY:
            save_uv_gs_2_ply( torch.from_numpy(uvgs_map).permute(0,1,3,2) , ply_path=f'{out_path}/{folder}.ply')
        
        

    
 