import copy

import numpy as np
import open3d as o3d


def paint_uniform_color(pcd, color):
    colors = np.array([color for _ in range(len(pcd.points))])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def display(pcds, colors=None):
    if colors is not None:
        for pcd, color in zip(pcds, colors):
            paint_uniform_color(pcd, color)
    o3d.visualization.draw_geometries(pcds)


def scale_point_cloud(pcd, scale_factor):
    pcd_scaled = copy.deepcopy(pcd)
    pcd_scaled = pcd_scaled.scale(scale_factor)
    return pcd_scaled


def translate_point_cloud(pcd, translation):
    pcd_translated = copy.deepcopy(pcd)
    pcd_translated = pcd_translated.translate(translation)
    return pcd_translated
