#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Nov 30, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Handle OpenCV import
from import_cv2 import *

# External modules
import numpy as np
from pathlib import Path

# ROS modules
import rospy
from cv_bridge import CvBridge, CvBridgeError
from tf2_msgs.msg import TFMessage
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Vector3, Quaternion, Transform, TransformStamped

CV_BRIDGE = CvBridge()


def ros_time_from_nsecs(timestamp):
    return rospy.Time.from_sec(timestamp * 1e-9)


def make_image_message(img, header, coding='bgr8'):
    # Convert grayscale to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Create image using CV bridge
    msg = CV_BRIDGE.cv2_to_imgmsg(img, coding)
    msg.header = header
    return msg


def make_pointcloud_message(header, plydata):
    # Extract data from plydata
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    intensity = plydata['vertex']['intensity']
    laser_number = plydata['vertex']['laser_number']

    # Prepare point cloud data
    points = list(zip(x, y, z, intensity, laser_number))
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('intensity', 12, PointField.UINT8, 1),
              PointField('laser_number', 13, PointField.UINT16, 1)]

    # Create point cloud message
    msg = point_cloud2.create_cloud(header, fields, points)
    return msg


def make_vector3_message(vector):
    msg = Vector3()
    msg.x = vector[0]
    msg.y = vector[1]
    msg.z = vector[2]
    return msg


def make_quaternion_message(coefficients, to_xyzw=True):
    # Convert WXYZ to XYZW is flag set
    if to_xyzw: coefficients = np.roll(coefficients, -1)
    msg = Quaternion()
    msg.x = coefficients[0]
    msg.y = coefficients[1]
    msg.z = coefficients[2]
    msg.w = coefficients[3]
    return msg


def make_transform_stamped_message(parent_frame, child_frame, transform):
    msg = TransformStamped()
    msg.header.frame_id = parent_frame
    # msg.header.stamp is filled later while writing to bag file
    msg.child_frame_id = child_frame
    msg.transform.translation = make_vector3_message(
        transform['translation'])
    msg.transform.rotation = make_quaternion_message(
        transform['rotation']['coefficients'])
    return msg
