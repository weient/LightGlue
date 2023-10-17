import json
import numpy as np
import open3d as o3d
from imageio import imread
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

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img',
                        help='Image texture in equirectangular format')
    parser.add_argument('--depth',
                        help='Depth map')
    parser.add_argument('--scale', default=0.001, type=float,
                        help='Rescale the depth map')
    #parser.add_argument('--crop_ratio', default=80/512, type=float,
    #                    help='Crop ratio for upper and lower part of the image')
    parser.add_argument('--crop_ratio', default=80/512, type=float,
                        help='Crop ratio for upper and lower part of the image')
    
    parser.add_argument('--crop_z_above', default=1.2, type=float,
                        help='Filter 3D point with z coordinate above')
    args = parser.parse_args()

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
    depth = np.expand_dims(depth, axis=2)
    print(depth.shape)
    #depth = imread(depth_path)[...,None].astype(np.float32) * args.scale
    
    # Project to 3d
    H, W = rgb.shape[:2]
    xyz = depth * get_uni_sphere_xyz(H, W)
    xyzrgb = np.concatenate([xyz, rgb/255.], 2)

    # Crop the image and flatten
    if args.crop_ratio > 0:
        assert args.crop_ratio < 1
        crop = int(H * args.crop_ratio)
        xyzrgb = xyzrgb[crop:-crop]
    xyzrgb = xyzrgb.reshape(-1, 6)

    # Crop in 3d
    xyzrgb = xyzrgb[xyzrgb[:,2] <= args.crop_z_above]

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])

    o3d.visualization.draw_geometries([
        pcd,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    ])

    o3d.io.write_point_cloud("/mnt/home_6T/public/weien/lightglue/new1.ply", pcd)