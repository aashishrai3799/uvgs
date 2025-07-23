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


import random
import numpy as np
from scipy.ndimage import distance_transform_edt


def cylindrical_unwrap(points, height=256, width=256, radius=1):
    """
    Unwrap a point cloud onto a cylindrical surface.

    Args:
    points (np.array): Input point cloud array of shape (N, 3), where N is the number of points.
    height (float): Height of the cylinder.
    radius (float): Radius of the cylinder.

    Returns:
    np.array: Unwrapped image of the cylindrical projection.
    """
    # Extract x, y, z coordinates
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]
    
    x = x1
    y = y1
    z = z1

    # Calculate cylindrical coordinates
    theta = np.arctan2(y, x)  # Angle around the cylinder
    h = z  # Height on the cylinder
    r = np.sqrt(x**2 + y**2)  # Radius - might be useful for radial correction

    # Normalize and scale theta and h to image coordinates
    theta = np.degrees(theta) + 180  # Convert angle to [0, 360] degrees
    theta_scaled = np.round((theta / 360) * width).astype(int)  # Map to image width
    h_scaled = np.round(((h - h.min()) / (h.max() - h.min())) * height).astype(int)  # Map to image height

    # Create an empty image
    image = np.zeros((int(height), width))

    # Fill image with values (for simplicity, using radius here, but you can use any scalar)
    for t, H, R in zip(theta_scaled, h_scaled, r):
        if 0 <= H < height and 0 <= t < width:
            # image[H, t] = max(image[H, t], R)  # Maximum radius projection
            image[H, t] = radius

    return image

def cylindrical_unwrap_xyz(points, height=256, width=256, radius=1):
    """
    Unwrap a point cloud onto a cylindrical surface.

    Args:
    points (np.array): Input point cloud array of shape (N, 3), where N is the number of points.
    height (float): Height of the cylinder.
    radius (float): Radius of the cylinder.

    Returns:
    np.array: Unwrapped image of the cylindrical projection.
    """
    # Extract x, y, z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Calculate cylindrical coordinates
    theta = np.arctan2(y, x)  # Angle around the cylinder
    h = z  # Height on the cylinder
    r = np.sqrt(x**2 + y**2)  # Radius - might be useful for radial correction

    # Normalize and scale theta and h to image coordinates
    theta = np.degrees(theta) + 180  # Convert angle to [0, 360] degrees
    theta_scaled = np.round((theta / 360) * width).astype(int)  # Map to image width
    h_scaled = np.round(((h - h.min()) / (h.max() - h.min())) * height).astype(int)  # Map to image height

    # Create an empty image
    image = np.zeros((int(height), width, 3))

    # Fill image with values (for simplicity, using radius here, but you can use any scalar)
    for t, H, xyz in zip(theta_scaled, h_scaled, points):
        if 0 <= H < height and 0 <= t < width:
            # image[H, t] = max(image[H, t], R)  # Maximum radius projection
            image[H, t, :] = xyz

    return image


def invert_cylindrical_projection(image, width, radius):
    """
    Invert a cylindrical projection image back into a 3D point cloud.

    Args:
    image (np.array): The 3D image of the cylindrical projection with dimensions (height, width, 3).
    width (int): The original width of the projection image in pixels.
    radius (float): The radius of the cylinder used in the original projection.

    Returns:
    np.array: A 3D point cloud restored from the cylindrical image.
    """
    height, img_width, _ = image.shape
    points = []

    # Calculate angular increment based on the image width
    d_theta = 360 / width  # degrees

    for i in range(height):
        for j in range(img_width):
            if np.any(image[i, j, :] != 0):  # Check if the pixel contains a non-zero value (i.e., a point)
                # Reverse the angle adjustment and convert to radians
                theta = (d_theta * j - 180) * (np.pi / 180)  # Convert to radians

                # Fetch the x, y, z values directly from the pixel
                x, y, z = image[i, j]

                # Append the point to the list
                points.append([x, y, z])

    return np.array(points)


def spherical_unwrap_xyz(points, height=256, width=256, radius=1):
    """
    Unwrap a point cloud onto a spherical surface.

    Args:
    points (np.array): Input point cloud array of shape (N, 3), where N is the number of points.
    height (int): The vertical resolution of the output image.
    width (int): The horizontal resolution of the output image.
    radius (float): The radius of the sphere (not directly used in computation here but included for interface consistency).

    Returns:
    np.array: Unwrapped image of the spherical projection.
    """
    # Extract x, y, z coordinates
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]
    
    x = z1
    y = y1
    z = x1

    # Compute spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)  # Spherical radius
    theta = np.arctan2(y, x)  # Azimuthal angle
    phi = np.arccos(z / r)  # Polar angle, assuming r is never zero

    # Normalize and scale theta and phi to image coordinates
    theta = np.degrees(theta) + 180  # Convert angle to [0, 360] degrees
    phi = np.degrees(phi)  # Convert polar angle to degrees
    theta_scaled = np.round((theta / 360) * width).astype(int)  # Map to image width
    phi_scaled = np.round((phi / 180) * height).astype(int)  # Map to image height

    # Create an empty image
    image = np.zeros((height, width, 3))
    counter = 0

    # Fill image with original xyz values
    for t, P, xyz in zip(theta_scaled, phi_scaled, points):
        if 0 <= P < height and 0 <= t < width:
            if image[P, t, :].all() == 0:
                image[P, t, :] = 1
            else:
                counter+=1

    print('Overlapping Points:', counter)
    return image


def spherical_unwrap_opacity(points, opacity, height=256, width=256):
    """
    Unwrap a point cloud onto a spherical surface.

    Args:
    points (np.array): Input point cloud array of shape (N, 3), where N is the number of points.
    height (int): The vertical resolution of the output image.
    width (int): The horizontal resolution of the output image.
    radius (float): The radius of the sphere (not directly used in computation here but included for interface consistency).

    Returns:
    np.array: Unwrapped image of the spherical projection.
    """
    # Extract x, y, z coordinates
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]
    
    x = z1
    y = y1
    z = x1

    # Compute spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)  # Spherical radius
    theta = np.arctan2(y, x)  # Azimuthal angle
    phi = np.arccos(z / r)  # Polar angle, assuming r is never zero

    # Normalize and scale theta and phi to image coordinates
    theta = np.degrees(theta) + 180  # Convert angle to [0, 360] degrees
    phi = np.degrees(phi)  # Convert polar angle to degrees
    theta_scaled = np.round((theta / 360) * width).astype(int)  # Map to image width
    phi_scaled = np.round((phi / 180) * height).astype(int)  # Map to image height

    # Create an empty image
    image = np.zeros((height, width, 3))
    opacity_image = np.zeros((height, width, 1))

    # Fill image with original xyz values
    for ind, t, P, xyz, opac in zip(range(len(opacity)), theta_scaled, phi_scaled, points, opacity):
        if 0 <= P < height and 0 <= t < width:
            if image[P, t, :].all() == 0:
                image[P, t, :] = ind
                opacity_image[P, t, :] = opac
            else:
                if opac > opacity_image[P, t, :]:
                    image[P, t, :] = ind

    return image


def spherical_unwrap_topK_opacity(points, opacity, height=256, width=256, K=4):
    """
    Unwrap a point cloud onto a spherical surface.

    Args:
    points (np.array): Input point cloud array of shape (N, 3), where N is the number of points.
    height (int): The vertical resolution of the output image.
    width (int): The horizontal resolution of the output image.
    radius (float): The radius of the sphere (not directly used in computation here but included for interface consistency).

    Returns:
    np.array: Unwrapped image of the spherical projection.
    """
    # Extract x, y, z coordinates
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]
    
    x = z1
    y = y1
    z = x1

    # Compute spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-12  # Spherical radius
    theta = np.arctan2(y, x)  # Azimuthal angle
    phi = np.arccos(z / r)  # Polar angle, assuming r is never zero

    # Normalize and scale theta and phi to image coordinates
    theta = np.degrees(theta) + 180  # Convert angle to [0, 360] degrees
    phi = np.degrees(phi)  # Convert polar angle to degrees
    theta_scaled = np.round((theta / 360) * width).astype(int)  # Map to image width
    phi_scaled = np.round((phi / 180) * height).astype(int)  # Map to image height

    # Initialize the image arrays
    image = np.zeros((height, width, K), dtype=int)
    opacity_image = np.full((height, width, K), -np.inf)  # Initialize with a very low value for comparison

    # Fill the image with original xyz indices of top K opacity values per pixel
    for ind, t, P, opac in zip(range(len(opacity)), theta_scaled, phi_scaled, opacity):
        
        if 0 <= P < height and 0 <= t < width:
            # Get the current opacities at pixel (P, t)
            current_opacities = opacity_image[P, t, :]
            
            # Check if we have space for more values or if the current opacity is among the top K
            if -np.inf in current_opacities or opac > current_opacities.min():
                # Find the index of the lowest opacity in current_opacities
                min_index = current_opacities.argmin()
                
                # Replace the lowest opacity with the new higher (closer to zero) opacity value and index
                opacity_image[P, t, min_index] = opac
                image[P, t, min_index] = ind

    return image



def fast_spherical_unwrap_topK_opacity(points, opacity, height=256, width=256, K=4):
    """
    Projects a point cloud onto a spherical surface and records
    the indices of the top-K points per pixel based on `opacity`.

    Args:
        points  (np.array): (N, 3) array with XYZ for N points.
        opacity (np.array): (N,) array of opacity values.
        height        (int): height of the output UV map.
        width         (int): width of the output UV map.
        K             (int): number of top opacity points to keep per pixel.

    Returns:
        np.array: (height, width, K) array of point indices. 
                  If a pixel has fewer than K points, 
                  the remaining indices will be 0.
    """
    # Ensure inputs are NumPy arrays
    points = np.asarray(points)
    opacity = np.asarray(opacity).ravel()  # flatten just in case

    # Unpack x, y, z
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]

    # Remap coordinates as in the original code
    x = z1
    y = y1
    z = x1

    # Compute spherical coordinates
    r     = np.sqrt(x**2 + y**2 + z**2) + 1e-12
    theta = np.arctan2(y, x)              # range [-pi, pi]
    phi   = np.arccos(z / r)              # range [0, pi]

    # Convert to degrees & shift
    theta_deg = np.degrees(theta) + 180.0  # [0..360]
    phi_deg   = np.degrees(phi)            # [0..180]

    # Scale angles to pixel coordinates
    theta_scaled = np.rint((theta_deg / 360.0) * width).astype(int)
    phi_scaled   = np.rint((phi_deg   / 180.0) * height).astype(int)

    # Filter out out-of-bounds pixels
    in_bounds = (
        (theta_scaled >= 0) & (theta_scaled < width) &
        (phi_scaled   >= 0) & (phi_scaled   < height)
    )
    if not np.any(in_bounds):
        return np.zeros((height, width, K), dtype=int)

    # Keep only valid points
    valid_theta = theta_scaled[in_bounds]
    valid_phi   = phi_scaled[in_bounds]
    valid_opac  = opacity[in_bounds]
    valid_idx   = np.nonzero(in_bounds)[0]  # original point indices

    # Flatten row/col into a single integer pixel index
    pixel_idx = valid_phi * width + valid_theta

    # Sort by pixel_idx ascending, then by opacity descending
    # One trick is to sort by negative opacity for descending
    neg_opac   = -valid_opac
    # np.lexsort sorts by the last key first, so order is:
    #   1) pixel_idx ascending
    #   2) neg_opac ascending (⇒ opac descending)
    sort_order = np.lexsort((neg_opac, pixel_idx))

    pixel_idx_sorted  = pixel_idx[sort_order]
    valid_idx_sorted  = valid_idx[sort_order]
    # (We don't actually need the sorted opacities for the final output,
    #  only to determine which points are top K.)

    # Group by pixel_idx and pick the top-K from each group
    unique_pixels, start_positions, counts = np.unique(
        pixel_idx_sorted, 
        return_index=True, 
        return_counts=True
    )

    # Prepare output
    image = np.zeros((height, width, K), dtype=int)

    # For each pixel group, fill up to K points
    for pix, start_idx, count in zip(unique_pixels, start_positions, counts):
        # row, col
        row = pix // width
        col = pix % width

        # Number of points to assign
        num_to_place = min(count, K)
        # The first 'num_to_place' points in this group are top opacities
        top_indices = valid_idx_sorted[start_idx : start_idx + num_to_place]
        image[row, col, :num_to_place] = top_indices

    return image




def spherical_unwrap_fill_channels(points, height=256, width=256, K=4):
    """
    Unwrap a point cloud onto a spherical surface and fill each pixel with the 
    indices of points in the order they arrive until K channels per pixel are filled.
    If a pixel receives fewer than K points, the remaining channels remain zero.
    
    Args:
        points (np.array): Input point cloud of shape (N, 3), where each row is (x, y, z).
        opacity (np.array): Opacity values for each point (unused here but kept for interface consistency).
        height (int): Vertical resolution of the output UV map.
        width (int): Horizontal resolution of the output UV map.
        K (int): Maximum number of points to store per pixel.
    
    Returns:
        np.array: A 3D array of shape (height, width, K) containing point indices.
                Pixels with fewer than K points will have remaining channels set to 0.
    """
    # Extract coordinates
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]
    
    # Remap coordinates as in the original code: x = z1, y = y1, z = x1
    x = z1
    y = y1
    z = x1

    # Compute spherical coordinates (r, theta, phi)
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-12  # Add epsilon to avoid division by zero
    theta = np.arctan2(y, x)                # Azimuthal angle in radians, range [-pi, pi]
    phi = np.arccos(z / r)                  # Polar angle in radians, range [0, pi]

    # Convert angles to degrees and shift so that theta is in [0, 360] and phi in [0, 180]
    theta_deg = np.degrees(theta) + 180
    phi_deg = np.degrees(phi)

    # Scale angles to pixel coordinates
    theta_scaled = np.round((theta_deg / 360) * width).astype(int)
    phi_scaled = np.round((phi_deg / 180) * height).astype(int)

    # Initialize the output array.
    # "image" will store point indices. Channels not filled remain 0.
    image = np.zeros((height, width, K), dtype=int)
    # A boolean array to track which channels at each pixel are filled.
    filled = np.zeros((height, width, K), dtype=bool)
    # counter = 0
    

    # Loop over each point and assign its index to the first available channel in its pixel.
    for ind, t, P in zip(range(len(points)), theta_scaled, phi_scaled):
        if 0 <= P < height and 0 <= t < width:
            empty_channels = np.where(~filled[P, t, :])[0]
            if empty_channels.size > 0:
                channel = empty_channels[0]
                image[P, t, channel] = ind
                filled[P, t, channel] = True
                # counter+=1
    # print(counter)

    return image


def fast_spherical_unwrap_fill_channels(points, height=256, width=256, K=4):
    """
    Unwrap a point cloud onto a spherical surface and fill each pixel with the 
    indices of points in the order they arrive until K channels per pixel are filled.
    If a pixel receives fewer than K points, the remaining channels remain zero.
    
    Args:
        points (np.array): (N, 3) array of XYZ coordinates.
        height (int): Output UV map height.
        width (int): Output UV map width.
        K (int): Number of channels to fill per pixel.
    
    Returns:
        np.array: (height, width, K) array of point indices. Unfilled = 0.
    """

    # Unpack coordinates
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]

    # Remap as per original code: x=z1, y=y1, z=x1
    x = z1
    y = y1
    z = x1

    # Compute spherical coords
    # r = radius, theta = azimuth, phi = polar angle
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-12
    theta = np.arctan2(y, x)          # range [-pi, pi]
    phi = np.arccos(z / r)            # range [0, pi]

    # Convert to degrees & shift
    theta_deg = np.degrees(theta) + 180.0  # [0..360]
    phi_deg   = np.degrees(phi)            # [0..180]

    # Scale angles to pixel coordinates
    # round(...) can produce values outside [0, width-1/ height-1], so we'll clamp below
    theta_scaled = (theta_deg / 360.0) * width
    phi_scaled   = (phi_deg   / 180.0) * height

    # Round to int
    theta_scaled = np.rint(theta_scaled).astype(int)
    phi_scaled   = np.rint(phi_scaled).astype(int)

    # Mask out-of-bounds
    valid_mask = (
        (theta_scaled >= 0) & (theta_scaled < width) &
        (phi_scaled   >= 0) & (phi_scaled   < height)
    )
    if not np.any(valid_mask):
        # If no points are in-bounds, just return zeros
        return np.zeros((height, width, K), dtype=int)

    # Keep only valid points
    valid_theta = theta_scaled[valid_mask]
    valid_phi   = phi_scaled[valid_mask]
    # These are the original indices of each valid point
    valid_indices = np.nonzero(valid_mask)[0]

    # Combine row/col into a single pixel index: pixel = row*width + col
    pixel_idx = (valid_phi * width) + valid_theta

    # We want the first K points in the order they arrive,
    # so we use a STABLE sort by pixel index
    order = np.argsort(pixel_idx, kind='stable')
    pixel_idx_sorted = pixel_idx[order]
    valid_indices_sorted = valid_indices[order]

    # Figure out how many points each unique pixel has
    #  - unique_pixels : sorted array of pixel IDs
    #  - start_inds[i] : start index in 'pixel_idx_sorted' where pixel i's data begins
    #  - counts[i]     : number of points for pixel i
    unique_pixels, start_inds, counts = np.unique(
        pixel_idx_sorted,
        return_index=True,
        return_counts=True
    )

    # Prepare output
    image = np.zeros((height, width, K), dtype=int)

    # Assign the first K points from each pixel group
    # upix is the single integer pixel ID; row = upix // width, col = upix % width
    for upix, st, cnt in zip(unique_pixels, start_inds, counts):
        row = upix // width
        col = upix % width
        # how many points can we place at this pixel?
        num_to_place = min(cnt, K)
        # the original point indices
        these_points = valid_indices_sorted[st : st + num_to_place]
        image[row, col, :num_to_place] = these_points

    return image



def spherical_unwrap_count_K(points, uv_size=1024):
    """
    Unwrap a point cloud onto a spherical surface.

    Args:
    points (np.array): Input point cloud array of shape (N, 3), where N is the number of points.
    height (int): The vertical resolution of the output image.
    width (int): The horizontal resolution of the output image.
    radius (float): The radius of the sphere (not directly used in computation here but included for interface consistency).

    Returns:
    np.array: Unwrapped image of the spherical projection.
    """
    
    height, width = uv_size, uv_size
    
    # Extract x, y, z coordinates
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]
    
    x = z1
    y = y1
    z = x1

    # Compute spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-12  # Spherical radius
    theta = np.arctan2(y, x)  # Azimuthal angle
    phi = np.arccos(z / r)  # Polar angle, assuming r is never zero

    # Normalize and scale theta and phi to image coordinates
    theta = np.degrees(theta) + 180  # Convert angle to [0, 360] degrees
    phi = np.degrees(phi)  # Convert polar angle to degrees
    theta_scaled = np.round((theta / 360) * width).astype(int)  # Map to image width
    phi_scaled = np.round((phi / 180) * height).astype(int)  # Map to image height

    # Initialize the image arrays
    counter_image = np.zeros((height, width), dtype=int)  # Initialize with a very low value for comparison

    # Fill the image with original xyz indices of top K opacity values per pixel
    for ind, t, P in zip(range(len(points)), theta_scaled, phi_scaled):
        if 0 <= P < height and 0 <= t < width:
            counter_image[P, t] += 1

    return counter_image


# STANDARD EQUIRECTANGULAR PROJECTION -x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

'''
Below is a explanation of how to convert the existing “spherical projection” approach (used in the UVGS paper) into a standard equirectangular projection:

---

1. Coordinate Definitions  
   - We compute a point's spherical radius `r` as `sqrt(x^2 + y^2 + z^2)`.  
   - We define longitude `lon` = `arctan2(y, x)` and latitude `lat` = `arcsin(z / r)`.  
     - `lon` ranges from -π to +π.  
     - `lat` ranges from -π/2 to +π/2.

2. Mapping to Image Coordinates  
   - Convert longitude `lon` in [-π, π] to a fraction `u` in [0, 1] by `(lon + π) / (2π)`.  
   - Convert latitude `lat` in [-π/2, π/2] to a fraction `v` in [0, 1] by `(lat + π/2) / π`.  
   - Map `u` and `v` to pixel coordinates for an image of size (height, width):  
     - `col = round(u * (width - 1))`  
     - `row = round(v * (height - 1))`

3. Implementation Outline  
   1. Extract `x, y, z` from the input array `points`.  
   2. Compute `r = sqrt(x^2 + y^2 + z^2)` (with a small epsilon to avoid division by zero).  
   3. Compute `lon = arctan2(y, x)` and `lat = arcsin(z / r)`.  
   4. Convert `lon` and `lat` to normalized coordinates `u` and `v` in [0,1].  
   5. Scale these to integer pixel coordinates.  
   6. Initialize an output image (for instance, a 3D array with dimensions `(height, width, K)`) and a parallel array for storing the top K opacities.  
   7. For each point, place its index in the correct pixel location if its opacity is among the top K for that pixel.

4. Top-K Opacity Logic  
   - We maintain an array `opacity_image` to store the top K opacity values per pixel.  
   - Each new point's opacity is compared to the minimum of the current top K. If it's greater, we replace that minimum entry with the new point's index/opacity.

5. Standard Equirectangular vs. Original Spherical Projection  
   - The original code uses `phi = arccos(z / r)` (polar angle from the positive z-axis) and `theta = arctan2(y, x)` in [0, 2π]. That maps to a sphere such that the polar angle is along the vertical dimension.  
   - In an equirectangular projection, we typically interpret `lat` = arcsin(z / r) (range [-π/2, π/2]) as the vertical axis, and `lon` = arctan2(y, x) (range [-π, π]) as the horizontal axis.  

6. Orientation Notes  
   - With `lat = +π/2` at the top, the top row of the image corresponds to the “north pole” (z > 0).  
   - If you prefer a different orientation (e.g., the sphere's pole at the bottom), invert or shift the `lat` mapping accordingly.  

In summary, simply replace the existing spherical angle computations with longitude/latitude in equirectangular form and map them to `[0, width-1]` and `[0, height-1]` respectively.
'''

def equirectangular_unwrap_topK_opacity(points, opacity, height=512, width=512, K=4):
    """
    Unwrap a point cloud onto an equirectangular (lat-lon) image.

    Args:
        points (np.array): Input point cloud, shape (N,3), where N is #points.
                           Each row is (x, y, z).
        opacity (np.array): Opacity (or intensity) values per point, shape (N,).
        height (int): Vertical resolution of the output image.
        width (int):  Horizontal resolution of the output image.
        K (int): Number of top-opacity points to keep per pixel.

    Returns:
        np.array: An integer image array of shape (height, width, K),
                  containing point indices for the top K opacity values in each pixel.
                  Pixels with fewer than K points have some default fill (0 or no points).
    """
    # 1) Separate coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 2) Compute spherical radius and angles
    #    r = distance from origin
    #    lon = arctan2(y, x) in [-pi, pi]
    #    lat = arcsin(z / r) in [-pi/2, pi/2]
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-12  # small epsilon to avoid div-by-zero
    lon = np.arctan2(y, x)            # range: [-pi, pi]
    lat = np.arcsin(z / r)            # range: [-pi/2, pi/2]

    # 3) Convert angles to [0,1] range
    #    u in [0,1] for lon in [-pi, +pi]
    #    v in [0,1] for lat in [-pi/2, +pi/2]
    #    Then map to pixel coordinates:
    #      col = round(u * (width - 1))
    #      row = round(v * (height - 1))
    u = (lon + np.pi) / (2.0 * np.pi)          # map lon: [-pi, pi]   -> [0, 1]
    v = (lat + np.pi/2.0) / np.pi              # map lat: [-pi/2, pi/2] -> [0, 1]

    col = (u * (width  - 1)).round().astype(int)
    row = (v * (height - 1)).round().astype(int)

    # 4) Prepare output arrays
    #    image[p_row, p_col, :]   = top K indices
    #    opacity_image[p_row, p_col, :] = top K opacities
    image = np.zeros((height, width, K), dtype=int)
    opacity_image = np.full((height, width, K), -np.inf, dtype=float)

    # 5) Populate each pixel with up to K highest-opacity points
    for pt_idx, (r_i, c_i, opac) in enumerate(zip(row, col, opacity)):
        if 0 <= r_i < height and 0 <= c_i < width:
            current_opacities = opacity_image[r_i, c_i, :]
            # If there's space or this point has higher opacity than the lowest in top-K
            if -np.inf in current_opacities or opac > current_opacities.min():
                # Replace the lowest opacity
                min_index = current_opacities.argmin()
                opacity_image[r_i, c_i, min_index] = opac
                image[r_i, c_i, min_index] = pt_idx

    return image


def equirectangular_unwrap_random_opacity(points, opacity, height=512, width=512, K=4):
    """
    Unwrap a point cloud onto an equirectangular (lat-lon) image,
    but select points randomly in each pixel rather than taking the top K by opacity.

    Args:
        points (np.array): Input point cloud, shape (N,3), where N is #points.
                           Each row is (x, y, z).
        opacity (np.array): Opacity (or intensity) values per point, shape (N,).
                            (In this version, we do not sort by opacity anymore.)
        height (int): Vertical resolution of the output image.
        width (int):  Horizontal resolution of the output image.
        K (int): Maximum number of random points to keep per pixel.

    Returns:
        np.array:
            An integer image array of shape (height, width, K).
            Each pixel has up to K point indices chosen randomly from that pixel's list.
            If fewer than K points fall in a pixel, remaining slots are filled with 0.
    """
    # ------------------------------------------------------------------------
    # 1) Convert (x,y,z) to equirectangular pixel coordinates
    # ------------------------------------------------------------------------
    x = points[:, 0]
    y = points[:, 2]
    z = points[:, 1]

    r = np.sqrt(x**2 + y**2 + z**2) + 1e-12  # small epsilon to avoid div-by-zero
    lon = np.arctan2(y, x)            # range: [-pi, pi]
    lat = np.arcsin(z / r)            # range: [-pi/2, pi/2]

    # Normalize to [0,1], then map to pixel coordinates
    u = (lon + np.pi) / (2.0 * np.pi)    # map lon: [-pi, pi]   -> [0, 1]
    v = (lat + np.pi/2.0) / np.pi        # map lat: [-pi/2, pi/2] -> [0, 1]

    col = (u * (width  - 1)).round().astype(int)
    row = (v * (height - 1)).round().astype(int)

    # ------------------------------------------------------------------------
    # 2) Collect point indices for each pixel
    # ------------------------------------------------------------------------
    # Create a nested list to store all point indices that fall in (row, col)
    pixel_points = [[[] for _ in range(width)] for _ in range(height)]

    # Populate pixel_points[row][col] with the indices of points
    for idx, (r_i, c_i) in enumerate(zip(row, col)):
        if 0 <= r_i < height and 0 <= c_i < width:
            pixel_points[r_i][c_i].append(idx)

    # ------------------------------------------------------------------------
    # 3) Randomly select up to K points in each pixel
    # ------------------------------------------------------------------------
    # Create the output image array of shape (height, width, K)
    # Fill it with 0 by default (or use -1 if you prefer an "empty" marker).
    image = np.zeros((height, width, K), dtype=int)

    # For each pixel, pick K random indices from pixel_points
    for r_i in range(height):
        for c_i in range(width):
            # All points that fall in this pixel
            all_pts = pixel_points[r_i][c_i]

            if len(all_pts) > 0:
                # Randomly sample up to K points
                chosen = random.sample(all_pts, k=min(K, len(all_pts)))

                # Place them into the image array; any leftover remain 0
                for k_idx, pt_idx in enumerate(chosen):
                    image[r_i, c_i, k_idx] = pt_idx

    return image


# STANDARD EQUIRECTANGULAR PROJECTION ENDS HERE -x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-


def spherical_unwrap_pixel_shift(points, opacity, height=256, width=256, radius=1):
    """
    Unwrap a point cloud onto a spherical surface.

    Args:
    points (np.array): Input point cloud array of shape (N, 3), where N is the number of points.
    height (int): The vertical resolution of the output image.
    width (int): The horizontal resolution of the output image.
    radius (float): The radius of the sphere (not directly used in computation here but included for interface consistency).

    Returns:
    np.array: Unwrapped image of the spherical projection.
    """
    # Extract x, y, z coordinates
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]
    
    x = z1
    y = y1
    z = x1

    # Compute spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)  # Spherical radius
    theta = np.arctan2(y, x)  # Azimuthal angle
    phi = np.arccos(z / r)  # Polar angle, assuming r is never zero

    # Normalize and scale theta and phi to image coordinates
    theta = np.degrees(theta) + 180  # Convert angle to [0, 360] degrees
    phi = np.degrees(phi)  # Convert polar angle to degrees
    theta_scaled = np.round((theta / 360) * width).astype(int)  # Map to image width
    phi_scaled = np.round((phi / 180) * height).astype(int)  # Map to image height

    # Create an empty image
    image = np.zeros((height, width, 3))

    # Fill image with original xyz values
    for ind, t, P, xyz, opac in zip(range(len(opacity)), theta_scaled, phi_scaled, points, opacity):
        if 0 <= P < height and 0 <= t < width:
            if image[P, t, :].all() == 0:
                image[P, t, :] = ind
                # opacity_image[P, t, :] = opac
            else:
                timer = 0
                while(timer < (512-t-1)):
                    timer += 1
                    if image[P, t + timer, :].all() == 0:
                        image[P, t + timer, :] = ind
                        break

    # print('Overlapping Points:', counter)
    return image



def spherical_unwrap_rgb(points, rgb, height=256, width=256, radius=1):
    """
    Unwrap a point cloud onto a spherical surface.

    Args:
    points (np.array): Input point cloud array of shape (N, 3), where N is the number of points.
    height (int): The vertical resolution of the output image.
    width (int): The horizontal resolution of the output image.
    radius (float): The radius of the sphere (not directly used in computation here but included for interface consistency).

    Returns:
    np.array: Unwrapped image of the spherical projection.
    """
    # Extract x, y, z coordinates
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]
    
    x = z1
    y = y1
    z = x1

    # Compute spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)  # Spherical radius
    theta = np.arctan2(y, x)  # Azimuthal angle
    phi = np.arccos(z / r)  # Polar angle, assuming r is never zero

    # Normalize and scale theta and phi to image coordinates
    theta = np.degrees(theta) + 180  # Convert angle to [0, 360] degrees
    phi = np.degrees(phi)  # Convert polar angle to degrees
    theta_scaled = np.round((theta / 360) * width).astype(int)  # Map to image width
    phi_scaled = np.round((phi / 180) * height).astype(int)  # Map to image height

    # Create an empty image
    image = np.zeros((height, width, 3))

    # Fill image with original xyz values
    for t, P, xyz in zip(theta_scaled, phi_scaled, rgb):
        if 0 <= P < height and 0 <= t < width:
            image[P, t, :] = xyz

    return image



def cylindrical_unwrap_rgb(points, rgb, height=256, width=256, radius=10):
    """
    Unwrap a point cloud onto a cylindrical surface.

    Args:
    points (np.array): Input point cloud array of shape (N, 3), where N is the number of points.
    height (float): Height of the cylinder.
    radius (float): Radius of the cylinder.

    Returns:
    np.array: Unwrapped image of the cylindrical projection.
    """
    # Extract x, y, z coordinates
    x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]
    
    x = z1
    y = y1
    z = x1

    # Calculate cylindrical coordinates
    theta = np.arctan2(y, x)  # Angle around the cylinder
    h = z  # Height on the cylinder
    r = np.sqrt(x**2 + y**2)  # Radius - might be useful for radial correction

    # Normalize and scale theta and h to image coordinates
    theta = np.degrees(theta) + 180  # Convert angle to [0, 360] degrees
    theta_scaled = np.round((theta / 360) * width).astype(int)  # Map to image width
    h_scaled = np.round(((h - h.min()) / (h.max() - h.min())) * height).astype(int)  # Map to image height

    # Create an empty image
    image = np.zeros((int(height), width, 3))

    # Fill image with values (for simplicity, using radius here, but you can use any scalar)
    for t, H, rgb_col in zip(theta_scaled, h_scaled, rgb):
        if 0 <= H < height and 0 <= t < width:
            image[H, t, :] = rgb_col  # Maximum radius projection

    return image


def fill_missing_pixels(image):
    """
    Fill in missing pixels (assumed to be zero or a specific marker) using a distance-weighted interpolation.

    Args:
    image (np.array): Input 2D or 3D image with missing pixels marked by zero (or another unique value).

    Returns:
    np.array: Image with missing pixels filled in.
    """
    # Check if image is grayscale or RGB
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        is_color = False
    else:
        is_color = True
    
    # Create a mask of missing pixels
    missing_mask = (image[:,:,:3] == 0).all(axis=-1)
    
    # Compute the distance transform (inverse distance weights)
    distances = distance_transform_edt(~missing_mask)
    inv_distances = 1 / (distances + 1e-10)  # Avoid division by zero
    
    # Create output image initialized to the input image
    filled_image = np.copy(image)
    
    # For each channel
    for c in range(image.shape[-1]):
        # Temporary copy of the image channel
        temp_image = image[:,:,c]
        # Weighted sum of known pixel values
        weighted_sum = np.zeros_like(temp_image, dtype=float)
        # Sum of weights used for normalization
        weight_sum = np.zeros_like(temp_image, dtype=float)
        
        # Apply weights from nearby pixels within a window
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                
                shifted_weights = np.roll(np.roll(inv_distances, dx, axis=0), dy, axis=1)
                shifted_images = np.roll(np.roll(temp_image, dx, axis=0), dy, axis=1)
                
                valid_shift_mask = ~np.roll(np.roll(missing_mask, dx, axis=0), dy, axis=1)
                
                weighted_sum += shifted_images * shifted_weights * valid_shift_mask
                weight_sum += shifted_weights * valid_shift_mask
        
        # Normalize to compute the interpolated values
        filled_image[:,:,c] = np.where(missing_mask, weighted_sum / (weight_sum + 1e-10), temp_image)
    
    if not is_color:
        filled_image = filled_image[:,:,0]
    
    return filled_image


