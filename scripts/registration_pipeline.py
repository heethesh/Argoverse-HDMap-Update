import copy

import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy import stats

import utils_o3d as utils


def execute_global_registration(source_down, target_down, 
                                source_fpfh, target_fpfh, 
                                distance_threshold, num_iters, num_val_iters):
    print(':: RANSAC registration on downsampled point clouds.')
    print('   Distance threshold %.3f.' % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(num_iters, num_val_iters))
    return result


def remove_ground_plane(pcd, z_thresh=-2.7):
    cropped = copy.deepcopy(pcd)
    cropped_points = np.array(cropped.points)
    cropped_points = cropped_points[cropped_points[:, -1] > z_thresh]
    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(cropped_points)
    return pcd_final


def compute_features(pcd, voxel_size, normals_nn=100, features_nn=120, downsample=True):
    normals_radius  = voxel_size * 2
    features_radius = voxel_size * 4

    # Downsample the point cloud using Voxel grids 
    if downsample:
        print(':: Input size:', np.array(pcd.points).shape)
        print(':: Downsample with a voxel size %.3f.' % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)
        print(':: Downsample size', np.array(pcd.points).shape)
    else: pcd_down = copy.deepcopy(pcd)

    # Estimate normals  
    print(':: Estimate normal with search radius %.3f.' % normals_radius)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normals_radius, max_nn=normals_nn))

    # Compute FPFH features 
    print(':: Compute FPFH feature with search radius %.3f.' % features_radius)
    features = o3d.registration.compute_fpfh_feature(pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=features_radius, max_nn=features_nn))

    return pcd_down, features


def match_features(pcd0, pcd1, feature0, feature1, thresh=None, display=False):
    pcd0, pcd1 = copy.deepcopy(pcd0), copy.deepcopy(pcd1)
    print(':: Input size 0:', np.array(pcd0.points).shape)
    print(':: Input size 1:', np.array(pcd1.points).shape)
    print(':: Features size 0:', np.array(feature0.data).shape)
    print(':: Features size 1:', np.array(feature1.data).shape)

    utils.paint_uniform_color(pcd0, color=[0, 0, 1])
    utils.paint_uniform_color(pcd1, color=[0, 1, 0])

    scores, indices = [], []
    fpfh_tree = o3d.geometry.KDTreeFlann(feature1)
    for i in range(len(pcd0.points)):
        [_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
        scores.append(np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]]))
        indices.append([i, idx[0]])

    scores, indices = np.array(scores), np.array(indices)
    median = np.median(scores)
    if thresh is None: thresh = median
    inliers_idx = np.where(scores <= thresh)[0]
    pcd0_idx = indices[inliers_idx, 0]
    pcd1_idx = indices[inliers_idx, 1]

    print(':: Score stats: Min=%0.3f, Max=%0.3f, Median=%0.3f, N<Thresh=%d' % (
        np.min(scores), np.max(scores), median, len(inliers_idx)))

    if display:
        for i, j in zip(pcd0_idx, pcd1_idx):
            pcd0.colors[i] = [1, 0, 0]
            pcd1.colors[j] = [1, 0, 0]
        utils.display([pcd0, pcd1])

    return pcd0_idx, pcd1_idx


def estimate_scale(pcd0, pcd1, pcd0_idx, pcd1_idx, top_percent=1.0,
    ransac_iters=5000, sample_size=50):
    points0 = np.asarray(pcd0.points)[pcd0_idx]
    points1 = np.asarray(pcd1.points)[pcd1_idx]
    mean0 = np.mean(points0, axis=0)
    mean1 = np.mean(points1, axis=0)
    top_count = int(top_percent * len(pcd0_idx))
    assert top_count > sample_size, 'top_count <= sample_size'

    scales = []
    for i in tqdm(range(ransac_iters), desc='Scale Estimation RANSAC'):
        args = np.random.choice(top_count, sample_size, replace=False)
        points0_r = points0[args]
        points1_r = points1[args]

        score0 = np.sum((points0_r - mean0) ** 2, axis=1)
        score1 = np.sum((points1_r - mean1) ** 2, axis=1)
        scale = np.sqrt(np.mean(score1) / np.mean(score0))
        scales.append(scale)

    best_scale = stats.mode(scales)[0][0]
    print(':: Estimated scale:', best_scale)

    return best_scale


def run():
    voxel_size = 0.2
    dso_scale = 0.05

    pcd_lidar = o3d.io.read_point_cloud('../maps/scans/scan_050.pcd')
    pcd_lidar = remove_ground_plane(pcd_lidar)

    pcd_dso = o3d.io.read_point_cloud('../maps/scans/scan_055.pcd')
    pcd_dso = remove_ground_plane(pcd_dso)
    pcd_dso = utils.scale_point_cloud(pcd_dso, dso_scale) #.rotate([1, 2, 3])

    # pcd_lidar = o3d.io.read_point_cloud('../data/cloud_bin_0.pcd')
    # pcd_dso = o3d.io.read_point_cloud('../data/cloud_bin_1.pcd')

    # utils.display(pcds=[pcd_lidar, pcd_dso], colors=[[0, 0, 1], [0, 1, 0]])

    print('\nComputing FPFH features for lidar point cloud...')
    pcd_lidar_down, features_lidar = compute_features(pcd_lidar, voxel_size=voxel_size)

    print('\nComputing FPFH features for DSO point cloud...')
    pcd_dso_down, features_dso = compute_features(pcd_dso, voxel_size=voxel_size * (down_scale if down_scale < 1 else 1))

    print('\nMatching FPFH features...')
    pcd_lidar_idx, pcd_dso_idx = match_features(pcd_lidar_down, pcd_dso_down,
        features_lidar, features_dso, thresh=None)

    print('\nEstimating scale using matches...')
    estimate_scale(pcd_lidar_down, pcd_dso_down, pcd_lidar_idx, pcd_dso_idx)


if __name__ == '__main__':
    run()    
