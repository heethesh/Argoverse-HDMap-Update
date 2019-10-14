# To Do:
'''
1) Implement Lowe's ratio test
2) Implement multiple image matching pipeline
3) Implement FLANN based Approximate nearest neighbor matching pipeline 
4) Create separate classes for Stereo SFM and Monocular SFM  
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np 
import sys
import os
import json 

class Utils:
    
    # Check if folder exists. Otherwise make one
    def mkdirFolder(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    # Return true if folder is empty, else return false 
    def isFolderEmpty(self, path):
        dirContents = os.listdir(path)
        if len(dirContents) == 0:
            return True
        else:
            return False

    # Check if there are images in the source path. If yes, push all the image paths in list and return 
    # Returns: Sorted list of image paths
    def extractImagePaths(self, sourcePath):
        imagePaths= []
        filenames = sorted(os.listdir(sourcePath))

        # If the filename is an image, append the path of the image 
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
                imagePaths.append(os.path.join(sourcePath, filename))
        return imagePaths
    
    
    # Reads a .json file and returns the data from the .json file 
    def readJsonFile(self, jsonFilePath):
        with open(jsonFilePath) as json_file:
            data = json.load(json_file)
        return data


class CVUtils():
    
    # Read an image from given path 
    def readImage(self, imPath):
        img= cv2.imread(imPath)
        return img
    
    # Display and image with given window name 
    def displayImage(self, img, windowName= "Image"):
        cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(windowName, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Convert image to grayscale 
    def convertToGray(self, img):
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    # Detect features and descriptors from given image 
    def featureDetection(self, img, descriptorType, display= False):
        # Initialize feature detector object 
        if descriptorType == "orb":
            featureDetector= cv2.ORB_create()
        elif descriptorType == "brisk":
            featureDetector= cv2.BRISK_create()
        elif descriptorType == "sift":
            featureDetector= cv2.xfeatures2d.SIFT_create()
        elif descriptorType == "surf":
            featureDetector= cv2.xfeatures2d.SURF_create()
        
        # Convert image to grayscale 
        gray= self.convertToGray(img)

        # Find keypoints and descriptors 
        kp, des= featureDetector.detectAndCompute(img, None)

        # Display the image if flag
        if display:
            cv2.drawKeypoints(gray, kp, img)
            plt.imshow(img), plt.title(descriptorType), plt.show()
        
        return [kp, des]

    # Find the matches between two images 
    def featureMatching(self, img1, img2, descriptorType, displayMatches= False, displayKp= False):

        # Find features and descriptors from given images 
        [kp1, des1]= self.featureDetection(img1, descriptorType, displayKp)
        [kp2, des2]= self.featureDetection(img2, descriptorType, displayKp)

        if descriptorType == "sift" or descriptorType == "surf":
            matchCriteria= cv2.NORM_L2
        
        elif descriptorType == "brisk" or descriptorType == "orb":
            matchCriteria= cv2.NORM_HAMMING

        # create BFMatcher object
        bf = cv2.BFMatcher(matchCriteria, crossCheck= True)

        # Match descriptors.
        matches = bf.match(des1 , des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        # Display the matches if flag is true 
        if displayMatches:

            if(len(matches)> 100):
                print ("Number of matches more than 100. Drawing top 100 matches")
                img3= cv2.drawMatches(img1,kp1,img2,kp2, matches[:100], None, flags=2)
        
            else:
                print ("Drawing all matches")
                img3= cv2.drawMatches(img1,kp1,img2,kp2, matches, None, flags=2)

            plt.imshow(img3), plt.title(descriptorType), plt.show()
        
        return matches 



def main():
      
    # Utils object for OpenCV based tasks 
    cvUtils= CVUtils() 
    # Utils object for general tasks 
    utils= Utils()
    
    ################# Parsing input arguments ############
    # Path to Argoverse data folder 
    dataFolderPath= sys.argv[1]
    # Mode to run the SFM algorithm in: Stereo or Monocular 
    sfmMode= sys.argv[2]
    # Descriptor type for the matching process 
    descriptorType= sys.argv[3]
    # Path to calibration config json file 
    calibConfFilePath= dataFolderPath+ "/vehicle_calibration_info.json"
    # Path to city details json file 
    cityConfigFilePath= dataFolderPath+ "/city_info.json"

    ############### Loading image data from data folder ######

    cityName= utils.readJsonFile(cityConfigFilePath)['city_name']
    if(cityName!= "PIT"):
        print ("Data does not belong to Pittsburgh")
        return -1 
    else:
        print ("We have Pittsburgh data")

    if sfmMode == "stereo":
        print ("SFM in stereo mode")

        # Path to the stereo front left and front right folders
        stereoFrFolder=  dataFolderPath + "/stereo_front_right"
        stereoFlFolder= dataFolderPath + "/stereo_front_left" 

        # Paths to all the images in respective folders 
        stereoFrImgPaths=   utils.extractImagePaths(stereoFrFolder)
        stereoFlImgPaths=   utils.extractImagePaths(stereoFlFolder)

    imgR= cvUtils.readImage(stereoFrImgPaths[0])
    imgL= cvUtils.readImage(stereoFlImgPaths[0])
    
    # Extract matches from the first two images from the stereo folder 
    matches= cvUtils.featureMatching(imgR, imgL, descriptorType, displayMatches= True, displayKp= False)





  
if __name__== "__main__":
  main()

