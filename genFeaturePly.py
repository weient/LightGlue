import json
import numpy as np
import open3d as o3d
from imageio.v2 import imread
from PIL import Image
import cv2
import os
import argparse

def get_uni_sphere_xyz(H, W):
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = (i+0.5) / W * 2 * np.pi
    v = ((j+0.5) / H - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    sphere_xyz = np.stack([x, y, z], -1)
    return sphere_xyz


if __name__ == '__main__':
    # Rescale the depth map
    scale = 20
    # Crop ratio for upper and lower part of the image
    crop_ratio = 0
    # Filter 3D point with z coordinate above
    crop_z_above = 1.2
    crop_z_below = -0.7

    path = '/mnt/home_6T/public/weien/lightglue/'
    # img_path = path + 'rgb/point_p' + '000024' + '_view_equirectangular_domain_rgb.png'
    img_path = path + '0.png'
    # depth_path = path + 'mist/point_p' + '000022' + '_view_equirectangular_domain_mist.png'
    depth_path = path + 'depth0_0001.exr'
    
    # Reading rgb-d
    rgb = imread(img_path)
    rgb = rgb[:, :, :3]
    print(rgb.shape)
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    depth = np.expand_dims(depth, axis=2) * scale
    print(depth.shape)
    #depth = imread(depth_path)[...,None].astype(np.float32) * scale
    
    # Project to 3d
    H, W = rgb.shape[:2]
    xyz = depth * get_uni_sphere_xyz(H, W)
    xyzrgb = np.concatenate([xyz, rgb/255.], 2)

    # Crop the image and flatten
    if crop_ratio > 0:
        assert crop_ratio < 1
        crop = int(H * crop_ratio)
        xyzrgb = xyzrgb[crop:-crop]
    xyzrgb = xyzrgb.reshape(-1, 6)

    # Crop in 3d
    xyzrgb = xyzrgb[xyzrgb[:,2] <= crop_z_above]
    xyzrgb = xyzrgb[xyzrgb[:,2] >= crop_z_below]

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])

    o3d.visualization.draw_geometries([
        pcd,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    ])

    o3d.io.write_point_cloud("/home/shih/LightGlue/new1.ply", pcd)