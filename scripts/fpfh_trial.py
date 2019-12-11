# Load point clouds and display them 
import open3d as o3d
import numpy as np
import copy


def numpy_to_open3d(array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 4
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn= 30))

    radius_feature = voxel_size * 10
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn= 30))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def downsample_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down

def display_single_point_cloud(pcd, color_arr):
    pcd.paint_uniform_color(color_arr)
    o3d.visualization.draw_geometries([pcd])

# Color arr is from [0-1, 0-1, 0-1] (rgb)
def display_two_point_clouds(pcd1, pcd2, color_arr1, color_arr2):
    pcd1.paint_uniform_color(color_arr1)
    pcd2.paint_uniform_color(color_arr2)
    o3d.visualization.draw_geometries([pcd2, pcd1])

def pause_till_enter():
    input ("Press Enter to continue")

# Find mean and covariance of point cloud 
def get_mean_and_cov(pcd):
    [mean, cov]= pcd.compute_mean_and_covariance()
    return [mean, cov]

# Find centre of point cloud 
def get_cloud_centre(pcd):
    return (small_down.get_center())

# Find the number of points in the point cloud 
def num_points(pcd):
    points= np.asarray(pcd.points)
    num_points = points.shape[0]
    return num_points

# Extract 3D points for top N correspondences
def get_3d_correspondences(cloud1, cloud2, correspondence_set, top_N):
    # Extract indices for the top n correspondences
    correspondence_set_top = correspondence_set[0 : top_N, :]
    #Cloud 1 all 3d pts
    points1= np.asarray(cloud1.points)
    #Cloud 2 all 3d pts
    points2= np.asarray(cloud2.points)
    # Cloud 1 3D corresponding points 
    points1_corresp = points1[correspondence_set_top[: , 0], :]
    # Cloud 2 3D correspondning pts
    points2_corresp = points2[correspondence_set_top[: , 1], :]
    return [points1_corresp, points2_corresp]

def estimate_scale(mean1, mean2, points1_corresp, points2_corresp):
    return (np.sqrt(np.sum(np.square(points1_corresp - mean1)) / np.sum(np.square(points2_corresp - mean2))))


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    # Displaying both the point clouds
    display_two_point_clouds(small_down, large_down, [1., 0., 0.],[0., 1., 0.])
    source_temp.transform(transformation)
    display_two_point_clouds(source_temp, target_temp, [1., 0., 0.], [0., 1., 0.])

def draw_correspondences(corresp, pc):
    colors = [[0, 0, 1] for i in range(len(corresp))]
    points= numpy_to_open3d(corresp)
    point_cloud = o3d.geometry.PointCloud()
    #points = o3d.utility.Vector3dVector(corresp)

    #o3d.visualization.draw_geometries([pc])
    o3d.visualization.draw_geometries([points])

if __name__ == "__main__":
    
 
    #Loading two point clouds 
    small_cuboid = o3d.io.read_point_cloud("/home/aadityacr7/geometry_ws/src/argoverse_hdmap_updator/scripts/dso.pcd")
    large_cuboid = o3d.io.read_point_cloud("/home/aadityacr7/geometry_ws/src/argoverse_hdmap_updator/scripts/lidar.pcd")
    # Downsample two clouds 
    voxel_size = 0.5  # means 5cm for the dataset

    # Scaling the point cloud 
    # small_cuboid.scale(15)

    # Downsample clouds, estimate normals, fpfh features 
    [small_down, small_fpfh]= preprocess_point_cloud(small_cuboid, voxel_size)
    [large_down, large_fpfh] = preprocess_point_cloud(large_cuboid, voxel_size)
 
    # Execute global registration using Open3D
    result = execute_global_registration(small_down, large_down, small_fpfh,  large_fpfh, voxel_size)
    fitness= result.fitness
    inlier_rmse =  result.inlier_rmse

    print ("Fitness: ", fitness)
    print ("Inlier RMSE: ", inlier_rmse)

    draw_registration_result(small_down, large_down, result.transformation)

    # # Extract indices for all the correspondences 
    # correspondences_indices = np.asarray(result.correspondence_set)
    # print (correspondences_indices.shape[0])
    # print (num_points(small_down) )

    # pause_till_enter()
    # # Sort according to top n correspondences
    # num_top_correspondences = int(np.floor( correspondences_indices.shape[0]))
    # Get 3D points for top N correspondences 
    #[points1_corresp, points2_corresp]= get_3d_correspondences(small_down, large_down, correspondences_indices, num_top_correspondences)
    
    #draw_correspondences(points1_corresp, small_down)

