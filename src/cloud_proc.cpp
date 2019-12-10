/*
To Do:
3) N scan based Lidar Mapping (ICP)
*/

// ROS Stuff 
#include "ros/ros.h"

// PCL ROS Stuff
#include "pcl_ros/point_cloud.h"

// Core PCL stuff
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/PCLPointCloud2.h>

// STL stuff 
#include <vector>

////////////////// Global PCL stuff 
// Original Point cloud in PCL format 
pcl::PCLPointCloud2::Ptr pcl_cloud(new pcl::PCLPointCloud2);
// Point cloud after voxel grid based downsampling 
pcl::PCLPointCloud2::Ptr downsampled_cloud(new pcl::PCLPointCloud2);
// Point cloud to store FPFH features point cloud 
pcl::PCLPointCloud2 fpfh_cloud;

// Vector to store FPFH features for each point cloud 
std::vector<pcl::PCLPointCloud2> fpfh_features_vec;

////////////////// Global Ros Params  
// Defining global variables for parameters 
double voxel_size = 0;
float fpfh_radius= 0;
int fpfh_k_nearest_neigh= 0; 
bool print_debug = true;

////////////////// Global ROS Publishers   
ros::Publisher original_cloud_pub;
ros::Publisher downsampled_cloud_pub;
ros::Publisher fpfh_features_pub;

void computeFPFH (const pcl::PCLPointCloud2::ConstPtr &input_cloud, pcl::PCLPointCloud2 &output,
                  int k, 
                  double radius)
{
  using namespace pcl;
  // Get point cloud normals from PCLPointCloud2
  PointCloud<PointNormal>::Ptr cloud_normals(new PointCloud<PointNormal>);
  fromPCLPointCloud2 (*input_cloud, *cloud_normals);
  // Estimate FPFH features 
  FPFHEstimation<PointNormal, PointNormal, FPFHSignature33> feature_estimator;
  feature_estimator.setInputCloud (cloud_normals);
  feature_estimator.setInputNormals (cloud_normals);
  feature_estimator.setSearchMethod (search::KdTree<PointNormal>::Ptr (new search::KdTree<PointNormal>));
  feature_estimator.setKSearch (k);
  feature_estimator.setRadiusSearch (radius);
  // Create FPFHS object and compute features 
  PointCloud<FPFHSignature33> fpfhs;
  feature_estimator.compute (fpfhs);
  // Convert data back
  pcl::PCLPointCloud2 output_fpfhs;
  toPCLPointCloud2 (fpfhs, output_fpfhs);
  concatenateFields (*input_cloud, output_fpfhs, output);
}

void voxel_grid_downsampling(const pcl::PCLPointCloud2::ConstPtr &input_cloud, double voxels_size)
{
    // Downsample the points
    pcl::VoxelGrid<pcl::PCLPointCloud2> downsample;
    downsample.setInputCloud(input_cloud);
    downsample.setLeafSize(voxels_size, voxels_size, voxels_size);
    downsample.filter(*downsampled_cloud);
}

void cloud_callback (const sensor_msgs::PointCloud2ConstPtr& ros_cloud){

    // Convert ROS Msg to PCL data and save as global pcl_cloud
    pcl_conversions::toPCL(*ros_cloud, *pcl_cloud);
    // Downsample the point cloud (stored in downsample_cloud)
    //voxel_grid_downsampling(pcl_cloud, voxel_size);
    // Find FPFH features and store in global fpfh_cloud
    //computeFPFH(downsampled_cloud, fpfh_cloud, fpfh_k_nearest_neigh, fpfh_radius);
    // Insert to FPFH features into global vector of features 
    //fpfh_features_vec.push_back(fpfh_cloud);
    // Publish the original point cloud 
    original_cloud_pub.publish(*pcl_cloud);
    // Publish the downsampled point cloud 
    //downsampled_cloud_pub.publish(*downsampled_cloud);
    // Publish FPFH features cloud 
    //fpfh_features_pub.publish(fpfh_cloud);
}

int main (int argc, char** argv) {
     
    // Initialize ROS Subscriber 
    ros::init (argc, argv, "cloud_sub");
    ros::NodeHandle nh("~");

    // Get params from launch file
    nh.getParam("voxel_size", voxel_size);
    nh.getParam("fpfh_k_nearest_neigh", fpfh_k_nearest_neigh);
    nh.getParam("fpfh_radius", fpfh_radius);
    //nh.getParam("print_debug ", print_debug );

    // Set defaults for parameters if the parameter is not passed 
    nh.param<double>("voxel_size", voxel_size, 0.01);
    nh.param<int>("fpfh_k_nearest_neigh", fpfh_k_nearest_neigh, 10);
    nh.param<float>("fpfh_radius", fpfh_radius, 0.05);
    //nh.param<bool>("print_debug ", print_debug, false);

    ros::Rate loop_rate(10);
    // Create a ROS Subscriber to subscribe to PCL messages
    ros::Subscriber sub;
    sub = nh.subscribe ("/argoverse/lidar/pointcloud", 1, cloud_callback);
    // Creating ROS Publishers for point cloud publishing at different phases 
    // Create a ROS publisher for the original LIDAR cloud 
    original_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar/original_cloud", 1);
    // Create a ROS publisher for the LIDAR cloud after downsampling using Voxel Grids 
    downsampled_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar/downsample_cloud", 1);
    // Create a ROS publisher for the FPFH histogram processed point cloud  
    fpfh_features_pub = nh.advertise<sensor_msgs::PointCloud2>("/lidar/fpfh_cloud", 1);
    ros::spin();
 }


