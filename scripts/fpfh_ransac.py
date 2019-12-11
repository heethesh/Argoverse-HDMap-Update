# Load point clouds and display them 
import open3d as o3d
import numpy as np
import copy
import utils_o3d


# To Do 
# 1. Check all the criteria for global registration 


if __name__ == "__main__":
    
    ############ Preparing initial point clouds 
    # Load original point cloud
    bunny_original = o3d.io.read_point_cloud("/home/aadityacr7/geometry_ws/src/argoverse_hdmap_updator/scripts/bunny.pcd")
    # Scale the point cloud
    bunny_scaled = utils_o3d.scale_point_cloud(bunny_original, 2)
    ################# Downsample the cloud and compute FPFH features
    # Settings for downsampling and finding normals and features 
    voxel_size = 0.2
    normals_radius  = voxel_size * 2
    features_radius = voxel_size * 4
    # Compute FPFH features for unscaled bunny 
    print ("Computing FPFH features for original cloud")
    [bunny_orig_down,bunny_orig_fpfh] = utils_o3d.compute_fpfh_features(bunny_original,voxel_size, 
                                                                        normals_radius, features_radius, 
                                                                        normals_nn=5, features_nn=5)                              
    print ("Computing FPFH features for scaled cloud")
    # Compute FPFH features for scaled bunny 
    [bunny_scaled_down,bunny_scaled_fpfh] = utils_o3d.compute_fpfh_features(bunny_scaled,voxel_size, 
                                                                        normals_radius, features_radius, 
                                                                        normals_nn=5, features_nn=5) 
    ####################### FPFH global registration first trial 
    # Point clouds after alignment should be below this threshold 
    distance_threshold = voxel_size * 0.5
    # Max number of iterations for convergence 
    num_iters = 10000
    # Number of iterations for validation process 
    num_val_iters = 500
    registration_trial1 = utils_o3d.execute_global_registration(source_down, target_down, 
                                                                       source_fpfh, target_fpfh, 
                                                                       distance_threshold)
    fitness_trial1= registration_trial1.fitness
    inlier_rmse_trial1 =  registration_trial1.inlier_rmse
    #Print results from trial 1 of registration process 
    print ("Fitness on trial 1: ", fitness_trial1)
    print ("Inlier RMSE on trial 1: ", inlier_rmse_trial1)
    ####################### Extract FPFH features from registration result  
    correspondences_indices = np.asarray(registration_trial1.correspondence_set)

    # Given all correspondences indices, 
    
    
    #utils_o3d.display_two_point_clouds(bunny_original, bunny_scaled, [1., 0., 0.],[0., 1., 0.] )
    #utils_o3d.display_single_point_cloud(bunny_original, [1., 0., 0.])