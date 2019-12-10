#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Dec 9, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Built-in modules
import os
import copy

# External modules
import numpy as np
from tqdm import tqdm

# Handle OpenCV import
from import_cv2 import *

# Open3D modules
import open3d
from open3d.open3d.geometry import voxel_down_sample

# ROS modules
import tf
import rospy
import rospkg
import tf2_ros
import ros_numpy
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

# Loacl python modules
import utils

# Global variables
rospack = rospkg.RosPack()


class LidarMapNode:
    def __init__(self, name='lidar_map_node', run=True):
        # Class variables
        self.name = name
        self.stop_flag = False
        
        self.count = 0
        self.point_clouds = []
        self.poses = []
        self.output_filename = os.path.join(rospack.get_path('argoverse_hdmap_updator'),
            'maps/scans/scan_%03d.pcd')

        self.map_frame = 'map'
        self.odom_frame = 'odom'
        self.velodyne_frame = 'velo_link'

        # Start node
        rospy.init_node(self.name, anonymous=True)
        rospy.loginfo('Current PID: [%d]' % os.getpid())
        self.tf = tf.TransformListener()
        self.tfb = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfb)

        # Handle params and topics
        input_pcl = rospy.get_param('~input_pcl', '/kitti/velodyne/pointcloud')
        output_pcl = rospy.get_param('~output_pcl', '/lidar_map/pointcloud')

        # Subscribe to topics
        self.input_pcl_sub = rospy.Subscriber(input_pcl, PointCloud2, self.callback, queue_size=1)

        # Publish output topics
        self.output_pcl_pub = rospy.Publisher(output_pcl, PointCloud2, queue_size=1)

        # Shutdown hook
        rospy.on_shutdown(self.shutdown_hook)

        # Run node
        if run: self.run()

    def run(self):
        # Keep python from exiting until this node is stopped
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo('Shutting down')

    def callback(self, pcl_msg):
        # Node stop has been requested
        if self.stop_flag: return

        # Transform point cloud to map frame
        # points_map = self.transform_point_cloud(pcl_msg)
        # points_map = utils.pointcloud_message_to_numpy(pcl_msg)

        pose = self.transform_point_cloud(pcl_msg)
        if pose is None: return
        self.poses.append(pose)

        # pcd = utils.pointcloud_message_to_open3d(pcl_msg)
        # open3d.io.write_point_cloud(self.output_filename % self.count, pcd)
        # self.count += 1
        # return

        # if points_map is None: return

        # Convert to lidar map
        self.point_clouds.append(pcl_msg)

        # Publish output
        # header = Header()
        # header.stamp = pcl_msg.header.stamp
        # header.frame_id = self.map_frame
        # msg = utils.make_pointcloud_message_open3d(header, self.lidar_map)
        # self.output_pcl_pub.publish(msg)

    def transform_point_cloud(self, msg):
        try:
            transform = self.tfb.lookup_transform(self.map_frame, self.velodyne_frame, rospy.Time())
            return transform
            # msg_trans = do_transform_cloud(msg, transform)
            # return utils.pointcloud_message_to_open3d(msg_trans)
        except tf2_ros.LookupException as e:
            print('Exception', e)
            return None

    def make_lidar_map(self):
        # Stack all point clouds
        points_all = np.vstack(self.point_clouds)

        # Voxel grid downsampling
        pcd = utils.numpy_to_open3d(points_all)
        self.lidar_map  = voxel_down_sample(pcd, voxel_size=0.05)
        print(len(self.lidar_map.points))

    def shutdown_hook(self):
        self.stop_flag = True
        print('%s shutdown' % self.name)

        count = 0
        for msg, transform in zip(self.point_clouds, self.poses):
            msg_trans = do_transform_cloud(msg, transform)
            pcd = utils.pointcloud_message_to_open3d(msg_trans)
            open3d.io.write_point_cloud(self.output_filename % count, pcd)
            count += 1

        # self.make_lidar_map()
        # open3d.io.write_point_cloud(self.output_filename, self.lidar_map)
        # print('Map saved to:', self.output_filename)


if __name__ == '__main__':
    # Run node
    node = LidarMapNode()
