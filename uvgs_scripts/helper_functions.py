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
import plyfile
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement



def plt_subplot(image1, image2, path):
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(1, 2)
    # Display each image in a subplot
    axs[0].imshow(image1)
    axs[0].axis("off")  # Turn off axis labels
    axs[1].imshow(image2)
    axs[1].axis("off")
    # Adjust spacing between subplots
    plt.tight_layout()
    # Show the combined image
    plt.show()
    plt.savefig(path, dpi=256)


def plt_subplot_all_attr(uvgs, path):
    """
    Create a 2x3 grid of subplots and display each image in a subplot.
    Parameters:
    path (str): The path to save the combined image.
    """

    xyz, scale, rot, rot2, opac, rgb = uvgs[:3].permute(1,2,0)*10, uvgs[3:6].permute(1,2,0)/2, uvgs[6:9].permute(1,2,0)*5, uvgs[6:10].permute(1,2,0), uvgs[10:11].permute(1,2,0), uvgs[11:].permute(1,2,0)*5

    # Create a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # Display each image in a subplot
    axs[0, 0].imshow(xyz)
    axs[0, 0].axis("off")  # Turn off axis labels
    axs[0, 1].imshow(scale)
    axs[0, 1].axis("off")
    axs[0, 2].imshow(rot)
    axs[0, 2].axis("off")
    axs[1, 2].imshow(rot2)
    axs[1, 2].axis("off")
    axs[1, 0].imshow(opac, cmap="gray")
    axs[1, 0].axis("off")
    axs[1, 1].imshow(rgb)

    # Adjust spacing between subplots
    plt.tight_layout()
    # Show the combined image
    plt.show()
    # Save the combined image
    plt.savefig(path, dpi=256)


def save_img(img, path):
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(path, dpi=128)
    plt.clf()

def random_sample_equal(input_features, num_samples_per_feature):
    """
    Randomly samples an equal number of points from each element of the input_features dictionary.
    Args:
    input_features (dict): A dictionary where each key is a feature name and each value is a numpy array of feature data.
    num_samples_per_feature (int): The number of samples to draw from each feature.
    Returns:
    dict: A dictionary containing the sampled data for each feature.
    """
    sampled_features = {}

    num_data_points = len(input_features['color'])

    if num_samples_per_feature <= num_data_points:
        # Randomly sample indices without replacement
        indices = np.random.choice(num_data_points, size=num_samples_per_feature, replace=False)

    # Iterate over each feature in the input_features dictionary
    for feature_name, feature_data in input_features.items():
        # Check if the requested number of samples is less than the available data
        if num_samples_per_feature > num_data_points:
            # If requested samples exceed available data, repeat all data cyclically
            repeat_count = (num_samples_per_feature // num_data_points) + 1

            if feature_name != 'opacity':
                repeated_data = feature_data.repeat((repeat_count, 1))[:num_samples_per_feature, :]
            else:
                repeated_data = feature_data.repeat(repeat_count)[:num_samples_per_feature]

            sampled_features[feature_name] = repeated_data

        else:
            # Use the indices to sample the feature data
            sampled_features[feature_name] = feature_data[indices]

    return sampled_features


def get_top_m_by_column(input_features, m, column_index=10):
    """
    Return the top M rows from a PyTorch tensor sorted by a specific column.

    Args:
    input_features (torch.Tensor): A tensor of shape (N, 14) containing feature data.
    m (int): The number of top rows to return based on the column values.
    column_index (int): The index of the column to sort by. Default is 10.

    Returns:
    torch.Tensor: A tensor containing the top M rows sorted by the specified column.
    """
    # Check if the requested column index is within the range of the tensor's second dimension
    if column_index >= input_features.shape[1]:
        raise IndexError("Column index out of range for the given tensor dimensions.")

    # Sorting by column 'column_index'
    # torch.sort returns both sorted values and indices; we only need indices here
    _, indices = torch.sort(input_features[:, column_index], descending=True)

    # Select the top M indices
    top_m_indices = indices[:m]

    # Gather the top M rows based on these indices
    top_m_rows = input_features[top_m_indices]

    return top_m_rows



def gs_reader(gs_path, return_shs=False):
    
    # Read PLY file
    with open(gs_path, 'rb') as f:
        ply_data = plyfile.PlyData.read(f)
        data = ply_data.elements[0].data

    # Extracting the 3DGS attributes
    position_offset = np.stack([data['x'], data['y'], data['z']], axis=-1)
    scale = np.stack([data['scale_0'], data['scale_1'] , data['scale_2'] ], axis=-1)
    rotation = np.stack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']], axis=-1)
    opacity = data['opacity']
    color = np.stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], axis=-1)  # Assuming f_dc has three components
    if return_shs:
        f_rest = np.stack([data[f'f_rest_{n}'] for n in range(str(data.dtype).count('f_rest'))], axis=-1)
        f_rest = torch.tensor(f_rest, dtype=torch.float32)
    
    # Convert numpy arrays to torch tensors
    position_offset = torch.tensor(position_offset, dtype=torch.float32)
    scale = torch.tensor(scale, dtype=torch.float32)
    rotation = torch.tensor(rotation, dtype=torch.float32)
    opacity = torch.tensor(opacity, dtype=torch.float32)
    color = torch.tensor(color, dtype=torch.float32)
    
    if return_shs:

        input_features =  {
            'position_offset': position_offset,
            'scale': scale,
            'rotation': rotation,
            'opacity': opacity,
            'color': color,
            'f_rest': f_rest
        }
    
    else:
       
       input_features =  {
            'position_offset': position_offset,
            'scale': scale,
            'rotation': rotation,
            'opacity': opacity,
            'color': color
        } 
    # input_features = random_sample_equal(input_features, num_samples_per_feature=NUM_POINTS**2)

    return input_features



def attribute_wise_norm(input_features):    # Scaling Function
    
    input_features['position_offset'] = input_features['position_offset']  / 2
    # input_features['position_offset'] = torch.nn.functional.normalize(input_features['position_offset'])
    input_features['scale'] = torch.exp(input_features['scale'])*2-1
    input_features['rotation'] = torch.nn.functional.normalize(input_features['rotation'])
    input_features['opacity'] = input_features['opacity'] / 20
    input_features['color'] = (input_features['color'] / 4.2).clip(-1,1)

    return input_features


def attribute_wise_unnorm(normalized_features):   # Scaling Function
    """
    Unnormalize the input features.
    Args:
        normalized_features (dict): The normalized features.
    Returns:
        dict: The unnormalized features.
    """

    # Unnormalize each feature
    normalized_features['position_offset'] = normalized_features['position_offset'] * 2
    normalized_features['scale'] = torch.log((normalized_features['scale']+1)/2)
    normalized_features['opacity'] = normalized_features['opacity'] * 20
    normalized_features['color'] = normalized_features['color'] * 4.2

    return normalized_features


# import OpenEXR
# import Imath


def load_depth_map(path):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(path)
    
    # Get the header
    header = exr_file.header()
    
    # Determine the data window
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    # Read the channels as floats
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ['R', 'G', 'B']  # Adjust if your EXR file uses different channels
    # Extract depth data (assuming it's stored in the 'R' channel)
    depth_str = exr_file.channel('R', pt)
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth.shape = (size[1], size[0])  # Reshape to image dimensions
    return process_depth_maps(depth.copy())
    
def process_depth_maps(depth_map):
    depth_map[depth_map>depth_map.mean()] = 0
    return depth_map


def feat_dict_to_device(input_features, device='cuda'):
    for feature_name, feature_data in input_features.items():
        input_features[feature_name] = feature_data.to(device)

    return input_features


def construct_list_of_attributes(features_dict, return_shs=False):
    
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

    # All channels except the 3 DC

    for i in range(features_dict['color'].shape[1]):
        l.append('f_dc_{}'.format(i))
        
    if return_shs:
        for i in range(features_dict['f_rest'].shape[1]):
            l.append('f_rest_{}'.format(i))

    l.append('opacity')

    for i in range(features_dict['scale'].shape[1]):
        l.append('scale_{}'.format(i))

    for i in range(features_dict['rotation'].shape[1]):
        l.append('rot_{}'.format(i))
        
    return l


def save_ply(features_dict, path, return_shs=False):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    normals = np.zeros_like(features_dict['position_offset'])

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dict, return_shs)]

    elements = np.empty(features_dict['position_offset'].shape[0], dtype=dtype_full)
    
    if return_shs:
        attributes = np.concatenate((features_dict['position_offset'], normals, 
                                     features_dict['color'], features_dict['f_rest'], 
                                     features_dict['opacity'], features_dict['scale'], 
                                     features_dict['rotation']), 
                                    axis=1)
    
    else:
        attributes = np.concatenate((features_dict['position_offset'], normals, 
                                     features_dict['color'], 
                                     features_dict['opacity'], features_dict['scale'], 
                                     features_dict['rotation']), 
                                    axis=1)
    
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)



def save_gs_to_file(xyzs, file_path, n_points=256):
    
    # Expects xyzs to be (512,512,14)

    save_ply( attribute_wise_unnorm( feat_dict_to_device( {
                    'position_offset': xyzs[:,:,:3].view(n_points*n_points,3),
                    'scale': xyzs[:,:,3:6].view(n_points*n_points,3),
                    'rotation': xyzs[:,:,6:10].view(n_points*n_points,4),
                    'opacity': xyzs[:,:,10:11].view(n_points*n_points,1),
                    'color': xyzs[:,:,11:].view(n_points*n_points,3)
                    }, 'cpu') ),
            path=file_path)


def save_uv_gs_2_ply(UV_gs_points, ply_path):
    
    mask = (UV_gs_points!=0).all(axis=-1)
    UV_gs_points_masked = UV_gs_points[mask]
    selected_gs = UV_gs_points_masked
    
    save_ply( feat_dict_to_device( {
            'position_offset': selected_gs[:,:3],
            'scale': selected_gs[:,3:6],
            'rotation': selected_gs[:,6:10],
            'opacity': selected_gs[:,10:11],
            'color': selected_gs[:,11:]
            }, 'cpu'),
    path=ply_path) 


def inverse_concat(input_data):
    """
    Reconstructs the original input_data_all tensor from the concatenated input_data.
    
    Args:
    input_data (torch.Tensor): The concatenated tensor of shape 
                               (batch_size, 26, height, width) where slices of 
                               the original input_data_all are concatenated.
    
    Returns:
    torch.Tensor: The reconstructed tensor of shape (batch_size, 14, height, width, 2).
    """
    batch_size, _, height, width = input_data.shape
    
    # Initialize the reconstructed tensor
    input_data_all = torch.zeros(batch_size, 14, height, width, 2, dtype=input_data.dtype, device=input_data.device)
    
    # Populate input_data_all with slices from input_data based on the original concatenation order
    input_data_all[:, :3, :, :, 0] = input_data[:, :3, :, :]
    input_data_all[:, :3, :, :, 1] = input_data[:, 3:6, :, :]
    
    input_data_all[:, 11:, :, :, 0] = input_data[:, 6:9, :, :]
    input_data_all[:, 11:, :, :, 1] = input_data[:, 9:12, :, :]
    
    input_data_all[:, 6:10, :, :, 0] = input_data[:, 12:16, :, :]
    input_data_all[:, 6:10, :, :, 1] = input_data[:, 16:20, :, :]
    
    input_data_all[:, 3:6, :, :, 0] = input_data[:, 20:23, :, :]
    input_data_all[:, 3:6, :, :, 1] = input_data[:, 23:, :, :]
    
    return input_data_all


def inverse_concat_single(input_data):
    """
    Reconstructs the original input_data_all tensor from the concatenated input_data.
    
    Args:
    input_data (torch.Tensor): The concatenated tensor of shape 
                               (batch_size, 26, height, width) where slices of 
                               the original input_data_all are concatenated.
    
    Returns:
    torch.Tensor: The reconstructed tensor of shape (batch_size, 14, height, width, 2).
    """
    batch_size, _, height, width = input_data.shape
    
    # Initialize the reconstructed tensor
    input_data_all = torch.zeros(batch_size, 14, height, width, dtype=input_data.dtype, device=input_data.device)
    
    # Populate input_data_all with slices from input_data based on the original concatenation order
    input_data_all[:, :3, :, :] = input_data[:, :3, :, :]    
    input_data_all[:, 11:, :, :] = input_data[:, 3:6, :, :]    
    input_data_all[:, 6:10, :, :] = input_data[:, 6:10, :, :]    
    input_data_all[:, 3:6, :, :] = input_data[:, 10:, :, :]
    
    return input_data_all


def feat_unnorm(uv_gs):
    
    uv_gs[:,:3] = uv_gs[:,:3]  * 2
    # uv_gs[:,3:6] = torch.log( ( uv_gs[:,3:6] + 1.000001 ) / 2 )
    # uv_gs[:,3:6] = torch.log( ( 1.000001 - uv_gs[:,3:6] ) / 2 )
    uv_gs[:,3:6] = ( uv_gs[:,3:6] * 5 ) - 5 
    uv_gs[:,6:10] = uv_gs[:,6:10] * 3
    uv_gs[:,10:11] = uv_gs[:,10:11] * 20
    uv_gs[:,11:14] = (uv_gs[:,11:14] * 4.2)
    
    return uv_gs

def save_uv_gs_2_ply_multilayer(UV_gs_points_topK, ply_path, K=2, return_shs=False):
    
    for ind in range(K):
        UV_gs_points = UV_gs_points_topK[:,:,:,ind]
        mask = (UV_gs_points!=0).any(axis=-1)
        UV_gs_points_masked = UV_gs_points[mask]
        if ind == 0:
            selected_gs = UV_gs_points_masked
        else:
            selected_gs = torch.concat([selected_gs, UV_gs_points_masked], dim=0)
            
    # selected_gs = feat_unnorm(selected_gs)
    
    if return_shs:
        save_ply( feat_dict_to_device( {
            'position_offset': selected_gs[:,:3],
            'scale': selected_gs[:,3:6],
            'rotation': selected_gs[:,6:10],
            'opacity': selected_gs[:,10:11],
            'color': selected_gs[:,11:14],
            'f_rest': selected_gs[:,14:]
            }, 'cpu'),
            path=ply_path, return_shs=True)
    
    else:
        save_ply( feat_dict_to_device( {
            'position_offset': selected_gs[:,:3],
            'scale': selected_gs[:,3:6],
            'rotation': selected_gs[:,6:10],
            'opacity': selected_gs[:,10:11],
            'color': selected_gs[:,11:14]
            }, 'cpu'),
            path=ply_path)


def get_gs_features_from_uv(UV_gs_points_topK, K=2):
    
    for ind in range(K):
        UV_gs_points = UV_gs_points_topK[:,:,:,ind]
        mask = (UV_gs_points!=0).all(axis=-1)
        UV_gs_points_masked = UV_gs_points[mask]
        if ind == 0:
            selected_gs = UV_gs_points_masked
        else:
            selected_gs = torch.concat([selected_gs, UV_gs_points_masked], dim=0)
            
    selected_gs = feat_unnorm(selected_gs)
    
    return selected_gs


def get_gs_features_from_uv_single(UV_gs_points_topK):
    
    UV_gs_points = UV_gs_points_topK
    mask = (UV_gs_points!=0).all(axis=-1)
    UV_gs_points_masked = UV_gs_points[mask]
    selected_gs = UV_gs_points_masked
    selected_gs = feat_unnorm(selected_gs)
    
    return selected_gs


def get_dict_from_gs(selected_gs):
    gs_features = {
        'position_offset': selected_gs[:,:3],
        'scale': selected_gs[:,3:6],
        'rotation': selected_gs[:,6:10],
        'opacity': selected_gs[:,10:11],
        'color': selected_gs[:,11:]
        }
    return gs_features


def prune_points_counter(prune_mask, prune_mask_UV, prune_mask_UV_K):
        
        _, prune_count = np.unique(prune_mask, return_counts=True)
        _, prune_count_UV = np.unique(prune_mask_UV, return_counts=True)
        _, prune_count_K = np.unique(prune_mask_UV_K, return_counts=True)
        
        if len(prune_count)==2:
            prune_true_count = prune_count[1]
        else:
            prune_true_count = 0
            
        if len(prune_count_UV)==2:
            prune_true_count_UV = prune_count_UV[1]
        else:
            prune_true_count_UV = 0
        
        if  len(prune_count_K)==2:
            prune_true_count_K = prune_count_K[1]
        else:
            prune_true_count_K = 0
        
        print(f"Opacity Pruning: {prune_true_count} || Spherical UV Pruning: {prune_true_count_UV} || Spherical Max K Pruning: {prune_true_count_K}")


def spherical_unwrap_count_K_prune_mask(points, max_K=48, uv_size=1024):
        """
        Given a point cloud (points of shape (N, 3)) as a torch.Tensor, compute its spherical UV mapping,
        and return a Boolean mask tensor of shape (N,) that is True for points that are kept and False for points that are pruned.
        
        For each UV pixel (defined via spherical mapping), if more than max_K points fall into that pixel, 
        randomly keep only max_K of them and mark the rest as False.
        
        Args:
            points (torch.Tensor): Input tensor of shape (N, 3) on any device.
            max_K (int): Maximum allowed points per UV pixel.
            uv_size (int): The output UV "image" is assumed square with shape (uv_size, uv_size).
        
        Returns:
            torch.Tensor: A Boolean tensor of shape (N,), where True means the point is kept.
        """
        
        # device = points.device
        height, width = uv_size, uv_size

        # Extract coordinates from points
        x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]
        
        # Re-map coordinates (consistent with the original code: x=z1, y=y1, z=x1)
        x = z1
        y = y1
        z = x1

        # Compute spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-12  # add epsilon to avoid division by zero
        theta = np.arctan2(y, x)                # Azimuthal angle in [-pi, pi]
        phi = np.arccos(z / r)                  # Polar angle in [0, pi]

        # Convert angles to degrees and map to image coordinates
        theta_deg = np.degrees(theta) + 180.0   # Map theta to [0, 360]
        phi_deg   = np.degrees(phi)             # Map phi to [0, 180]
        theta_scaled = np.round((theta_deg / 360.0) * width).astype(int)
        phi_scaled   = np.round((phi_deg   / 180.0) * height).astype(int)
        
        # Stack the pixel coordinates for grouping; each point is assigned to one UV pixel
        uv_pixels = np.stack((phi_scaled, theta_scaled), axis=1)  # shape (N, 2)
        
        # Get group labels for each unique UV pixel
        _, inverse_indices = np.unique(uv_pixels, axis=0, return_inverse=True)
        
        # Initialize the mask: all points are kept by default
        mask = np.ones(len(points), dtype=bool)
        
        # Process each group (i.e. each unique UV pixel)
        unique_groups = np.unique(inverse_indices)
        for group in unique_groups:
            group_indices = np.where(inverse_indices == group)[0]
            count = len(group_indices)
            if count > max_K:
                # Randomly choose max_K indices to keep from this group
                selected = np.random.choice(group_indices, size=max_K, replace=False)
                # Mark all points in the group as False, then set the selected ones to True
                mask[group_indices] = False
                mask[selected] = True

        return ~mask
    
def spherical_unwrap_valid_mask(points, uv_size):
    """
    For each 3D point, compute its spherical coordinates (theta, phi) and
    map them to image coordinates in a (height x width) "canvas."
    Return a boolean mask array indicating which points lie within [0, height) and [0, width).

    Args:
        points (np.array): Input point cloud array of shape (N, 3).
        opacity (np.array): Opacity per point (unused for validity check, but included for interface consistency).
        height (int): Vertical resolution of the implied image.
        width (int):  Horizontal resolution of the implied image.

    Returns:
        valid_mask (np.array of bool): Shape (N,), where valid_mask[i] = True if
                                       point i is within the image bounds, False otherwise.
    """
    
    height, width = uv_size, uv_size
    
    # 1) Extract x, y, z
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]
    
    # 2) Re-map to the orientation you used (z->x, y->y, x->z)
    #    x, y, z are the coordinates used for computing spherical coords:
    x = z1
    y = y1
    z = x1

    # 3) Compute spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-12  # small epsilon to avoid division by zero
    theta = np.arctan2(y, x)                # Azimuth angle in [-pi, pi]
    phi = np.arccos(z / r)                  # Polar angle in [0, pi]

    # 4) Convert to degrees and scale to image coordinates
    theta_deg = np.degrees(theta) + 180.0   # Range: [0, 360]
    phi_deg = np.degrees(phi)               # Range: [0, 180]

    theta_scaled = np.round((theta_deg / 360.0) * width).astype(int)
    phi_scaled   = np.round((phi_deg   / 180.0) * height).astype(int)

    # 5) Create an array to store validity (True/False)
    valid_mask = np.zeros(len(points), dtype=bool)

    # 6) For each point, check if it lies within the image dimension
    for i, (t, p) in enumerate(zip(theta_scaled, phi_scaled)):
        if 0 <= p < height and 0 <= t < width:
            valid_mask[i] = True
        else:
            valid_mask[i] = False

    return ~valid_mask

