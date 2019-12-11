import open3d as o3d
import numpy as np
import copy


############# Display Utils 
def display_single_point_cloud(pcd, color_arr):
    pcd.paint_uniform_color(color_arr)
    o3d.visualization.draw_geometries([pcd])

def display_two_point_clouds(pcd1, pcd2, color_arr1, color_arr2):
    pcd1.paint_uniform_color(color_arr1)
    pcd2.paint_uniform_color(color_arr2)
    o3d.visualization.draw_geometries([pcd1, pcd2])

##################### Transforming point clouds
def scale_point_cloud(pcd, scale_factor):
    pcd_scaled = copy.deepcopy(pcd)
    pcd_scaled = pcd_scaled.scale(scale_factor)
    return pcd_scaled

def translate_point_cloud(pcd, translation):
    pcd_translated = copy.deepcopy(pcd)
    pcd_translated = pcd_translated.translate(translation)
    return pcd_translated

#################### Computing FPFH features 
def compute_fpfh_features(pcd, voxel_size, normals_radius, features_radius, normals_nn=5, features_nn=5):
    # Downsample the point cloud using Voxel grids 
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # Estimate normals  
    print(":: Estimate normal with search radius %.3f." % normals_radius)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normals_radius, max_nn= normals_nn))
    # Compute FPFH features 
    print(":: Compute FPFH feature with search radius %.3f." % features_radius)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=features_radius, max_nn= features_nn))
    return [pcd_down, pcd_fpfh]


###################### Execute global registration using FPFH features

def execute_global_registration(source_down, target_down, 
                                source_fpfh, target_fpfh, 
                                distance_threshold, num_iters, num_val_iters):
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(num_iters, num_val_iters))
    return result