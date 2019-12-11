#include <iostream>
#include <vector>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/correspondence_sorting.h>

using namespace pcl;
using namespace pcl::io;

pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
pcl::visualization::PCLVisualizer ICPView("ICP Viewer");

// double computeCloudResolution(const pcl::PointCloud<PointXYZ>::ConstPtr &cloud)
// {
//     double res = 0.0;
//     int n_points = 0;
//     int nres;
//     std::vector<int> indices(2);
//     std::vector<float> sqr_distances(2);
//     pcl::search::KdTree<PointXYZ> tree;
//     tree.setInputCloud(cloud);

//     for (size_t i = 0; i < cloud->size(); ++i)
//     {
//         if (!pcl_isfinite((*cloud)[i].x))
//         {
//             continue;
//         }
//     //Considering the second neighbor since the first is the point itself.
//     nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
//     if (nres == 2)
//     {
//         res += sqrt(sqr_distances[1]);
//         ++n_points;
//     }
// }
// if (n_points != 0)
// {
//     res /= n_points;
// }
// return res;
// }

// Apply scale transform to the cloud 
// Eigen::Matrix4f transform2 = Eigen::Matrix4f::Identity(); 
// transform2 (0,0) = transform2 (0,0) * 2; 
// transform2 (1,1) = transform2 (1,1) * 2; 
// transform2 (2,2) = transform2 (2,2) * 2; 
// pcl::transformPointCloud (*target_cloud, *target_cloud2, transform2); 
// Compute model_resolution


/*
Compute normals given input cloud, search method and search radius 
*/
pcl::PointCloud<pcl::Normal>::Ptr compute_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                                                  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree, 
                                                  float normal_radius)
{
    // Compute the normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud);
    normalEstimation.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud< pcl::Normal>);
    normalEstimation.setRadiusSearch(normal_radius);
    normalEstimation.compute(*normals);
    return normals; 
}

/*
Compute FPFH features  
*/
pcl::PointCloud<pcl::FPFHSignature33>::Ptr compute_fpfh_features(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                                                                 pcl::PointCloud<pcl::Normal>::Ptr normals,
                                                                pcl::search::KdTree<pcl::PointXYZ>::Ptr tree, 
                                                                float fpfh_feature_radius
                                                                  )
{
    // Initialize object to store FPFH features
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>());
    // Initalize FPFH estimation object 
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    // Set parameters for FPFH estimation object 
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(0.2);
    fpfh.compute(*features);
    return features; 
}

int main(int, char** argv)
{
    ////////////////// Load clouds 
    // Load source cloud 
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new PointCloud<PointXYZ>());
    loadPCDFile("/home/aadityacr7/geometry_ws/src/argoverse_hdmap_updator/src/dso.pcd", *source_cloud);
    std::cout << "File 1 points: " << source_cloud->points.size() << std::endl;
    // Load target cloud 
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new PointCloud<PointXYZ>());
    loadPCDFile("/home/aadityacr7/geometry_ws/src/argoverse_hdmap_updator/src/lidar.pcd", *target_cloud);
    std::cout << "File 2 points: " << target_cloud->points.size() << std::endl;
    ////////////////// KD- Tree for correspondence search 
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ////////////////// Compute normals 
    // Source normals 
    pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud< pcl::Normal>);
    source_normals = compute_normals(source_cloud, tree, 0.02);
    // Target normals 
    pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud< pcl::Normal>);
    source_normals = compute_normals(target_cloud, tree, 0.02);
    //////////////////////// Compute FPFH features 
    // Source FPFH features 
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features(new pcl::PointCloud<pcl::FPFHSignature33>());
    source_features= compute_fpfh_features(source_cloud, source_normals, tree, 0.2);
    // Target FPFH features 
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features(new pcl::PointCloud<pcl::FPFHSignature33>());
    target_features= compute_fpfh_features(target_cloud, target_normals, tree, 0.2);
    //////////////////////// Estimate correspondences between FPFH features
    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> est;
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
    est.setInputSource(source_features);
    est.setInputTarget(target_features);
    est.determineCorrespondences(*correspondences);
    // Duplication rejection Duplicate
    pcl::CorrespondencesPtr correspondences_result_rej_one_to_one(new pcl::Correspondences());
    pcl::registration::CorrespondenceRejectorOneToOne corr_rej_one_to_one;
    corr_rej_one_to_one.setInputCorrespondences(correspondences);
    corr_rej_one_to_one.getCorrespondences(*correspondences_result_rej_one_to_one);

    std::cout << corr_rej_one_to_one.size() << std::endl; 
    // // Correspondance rejection RANSAC
    // Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    // pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> rejector_sac;
    // pcl::CorrespondencesPtr correspondences_filtered(new pcl::Correspondences());
    // rejector_sac.setInputSource(source_keypoints);
    // rejector_sac.setInputTarget(target_keypoints);
    // rejector_sac.setInlierThreshold(2.5); // distance in m, not the squared distance
    // rejector_sac.setMaximumIterations(1000000);
    // rejector_sac.setRefineModel(false);
    // rejector_sac.setInputCorrespondences(correspondences_result_rej_one_to_one);;
    // rejector_sac.getCorrespondences(*correspondences_filtered);
    // correspondences.swap(correspondences_filtered);
    // std::cout << correspondences->size() << " vs. " << correspondences_filtered->size() << std::endl;
    // transform = rejector_sac.getBestTransformation();   // Transformation Estimation method 1


    // // Transformation Estimation method 2
    // //pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
    // //transformation_estimation.estimateRigidTransformation(*source_keypoints, *target_keypoints, *correspondences, transform);
    // std::cout << "Estimated Transform:" << std::endl << transform << std::endl;

    // // / refinement transform source using transformation matrix ///////////////////////////////////////////////////////

    // pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_source(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr final_output(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::transformPointCloud(*source_cloud, *transformed_source, transform);
    // savePCDFileASCII("/home/aadityacr7/geometry_ws/src/argoverse_hdmap_updator/src/Transformed.pcd", (*transformed_source));


    // // viewer.setBackgroundColor (0, 0, 0);
    // viewer.setBackgroundColor(1, 1, 1);
    // viewer.resetCamera();
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_source_cloud(transformed_source, 150, 80, 80);
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_source_keypoints(source_keypoints, 255, 0, 0);

    // viewer.addPointCloud<pcl::PointXYZ>(transformed_source, handler_source_cloud, "source_cloud");
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source_cloud");
    // viewer.addPointCloud<pcl::PointXYZ>(source_keypoints, handler_source_keypoints, "source_keypoints");

    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_target_cloud(target_cloud, 80, 150, 80);
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_target_keypoints(target_keypoints, 0, 255, 0);

    // viewer.addPointCloud<pcl::PointXYZ>(target_cloud, handler_target_cloud, "target_cloud");
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target_cloud");
    // viewer.addPointCloud<pcl::PointXYZ>(target_keypoints, handler_target_keypoints, "target_keypoints");
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "target_keypoints");
    // viewer.addCorrespondences<pcl::PointXYZ>(source_keypoints, target_keypoints, *correspondences, 1, "correspondences");

    // pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    // icp.setInputSource(transformed_source);
    // icp.setInputTarget(target_cloud);
    // icp.align(*final_output);
    // std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    // icp.getFitnessScore() << std::endl;
    // std::cout << icp.getFinalTransformation() << std::endl;


    // ICPView.addPointCloud<pcl::PointXYZ>(final_output, handler_source_cloud, "Final_cloud");
    // ICPView.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "source_keypoints");
    // while (!viewer.wasStopped())
    // {

    //     viewer.spinOnce(100);
    //     boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    // }
    // while (!ICPView.wasStopped())
    // {

    //     ICPView.spinOnce(100);
    //     boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    // }
    /*
    // Setup the SHOT features
    typedef pcl::SHOT352 ShotFeature;
    pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, ShotFeature> shotEstimation;

    shotEstimation.setInputCloud(model);
    shotEstimation.setInputNormals(normals);
    shotEstimation.setIndices(keypoint_indices);

    // Use the same KdTree from the normal estimation
    shotEstimation.setSearchMethod(tree);
    pcl::PointCloud<ShotFeature>::Ptr shotFeatures(new pcl::PointCloud<ShotFeature>);
    //spinImageEstimation.setRadiusSearch (0.2);
    shotEstimation.setKSearch(10);

    // Actually compute the spin images
    shotEstimation.compute(*shotFeatures);
    std::cout << "SHOT output points.size (): " << shotFeatures->points.size() << std::endl;

    // Display and retrieve the SHOT descriptor for the first point.
    ShotFeature descriptor = shotFeatures->points[0];
    std::cout << descriptor << std::endl;
    */

    return 0;

}
