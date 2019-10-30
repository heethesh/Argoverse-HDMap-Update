# To Do:
'''
Must have features 
1) Filtering between matches (how do you quantify how good a match is)
2) Loading calibration parameters from the file 
3) Multiple image matching pipeline (aka choose good and bad images based on geometric checks)

Good to have features 
1) Implement Lowe's ratio test
2) Implement FLANN based Approximate nearest neighbor matching pipeline 
3) Create separate classes for Stereo SFM and Monocular SFM  
'''



# Argoverse data
#from argoverse.map_representation.map_api import ArgoverseMap
#from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
#from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader


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
        imagePaths = []
        filenames  = sorted(os.listdir(sourcePath))

        # If the filename is an image, append the path of the image 
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
                imagePaths.append(os.path.join(sourcePath, filename))
        return imagePaths
    
    
    # Reads a .json file and returns the data from the .json file 
    def readJsonFile(self, jsonFilePath):
        if os.path.exists(jsonFilePath):
            with open(jsonFilePath,'r') as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print("got %s on json.load()" % e)
        return data

class CVUtils():
    
    # Read an image from given path 
    def readImage(self, imPath):
        img = cv2.imread(imPath)
        return img
    
    # Display and image with given window name 
    def displayImage(self, img, windowName= "Image"):
        cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(windowName, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Convert image to grayscale 
    def convertToGray(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    # Detect features and descriptors from given image 
    def featureDetection(self, img, descriptorType, topNMatches, display= False):
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
        gray = self.convertToGray(img)

        # Find keypoints and descriptors 
        kp,des = featureDetector.detectAndCompute(img, None)

        # Display the image if flag
        if display:
            cv2.drawKeypoints(gray, kp, img)
            plt.imshow(img), plt.title(descriptorType), plt.show()
        
        return [kp, des]

    '''
    mask is (h, w) binary ndarray with ones where we want to keep feature points
    kp is (N, 2) ndarray
    '''   
    # Mask Pixel coordinates (to remove dynamic objects)
    def maskPixels(pts, mask):
         # Create a selection mask for all the keypoints to be kept where mask is 1.
        if mask is not None: select_mask = (mask[pts[:, 0], pts[:, 1]] == 1)
        else: select_mask = np.ones(len(pts))
        pts = pts[select_mask]
        return pts
       
    # Find the matches between two images 
    def featureMatching(self, img1, img2, descriptorType, topNMatches, mask1=None, mask2=None, displayMatches=False, displayKp=False):
        # Find features and descriptors from given images 
        [kp1, des1] = self.featureDetection(img1, descriptorType, displayKp)
        [kp2, des2] = self.featureDetection(img2, descriptorType, displayKp)

        # Find pixel correspondences as per matches 
        pts1 = np.asarray([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.asarray([kp2[m.trainIdx].pt for m in matches])
        
        #Mask feature points (for dynamic object removal)
        pts1Masked= maskPixels(pts1, mask1)
        pts2Masked= maskPixels(pts2, mask2)

        if descriptorType == "sift" or descriptorType == "surf":
            matchCriteria = cv2.NORM_L2
        
        elif descriptorType == "brisk" or descriptorType == "orb":
            matchCriteria = cv2.NORM_HAMMING

        # Match features.
        print('Matching features using KNN based matcher')
        indexParams = dict(algorithm=0, trees=5)
        searchParams = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexParams, searchParams)
        matches = flann.knnMatch(np.float32(des1), np.float32(des2), k=2)
        # Store all the good matches as per Lowe's ratio test.
        goodMatches = np.array([m for m, n in matches if m.distance < 0.7 * n.distance])
        # Sorted matches 
        sortedMatches= sorted(goodMatches, key = lambda x:x.distance)
        print ("Before taking top N matches: ", len(sortedMatches))
        # Removing the top N matches 
        matches= sortedMatches[:min(len(sortedMatches), topNMatches)]
        print ("After taking top N matches: ", len(matches))

        # Display the matches if flag is true 
        if displayMatches:
            print ("Drawing all matches")
            img3=  cv2.drawMatches(img1,kp1,img2,kp2, matches, None, flags=2)
            plt.imshow(img3), plt.title(descriptorType), plt.show()
        
        return [pts1Masked, pts2Masked, matches]

def main():
      
    # Utils object for OpenCV based tasks 
    cvUtils = CVUtils() 
    # Utils object for general tasks 
    utils = Utils()
    
    # Loading Argo Map API objects 
    #avm = ArgoverseMap()
    #avTrackerLoader = ArgoverseTrackingLoader('argoverse-tracking/')    #simply change to your local path of the data
    #avForecastingLoader = ArgoverseForecastingLoader('argoverse-forecasting/') #simply change to your local path of the data

    ################# Parsing input arguments ############
    # Path to Argoverse data folder 
    dataFolderPath = sys.argv[1]
    # Mode to run the SFM algorithm in: Stereo or Monocular 
    sfmMode = sys.argv[2]
    # Descriptor type for the matching process 
    descriptorType = sys.argv[3]
    # Top N matches 
    topNMatches= int(sys.argv[4])
    # Path to calibration config json file 
    calibConfFilePath = dataFolderPath+ "/vehicle_calibration_info.json"
    # Path to city details json file 
    cityConfigFilePath = dataFolderPath+ "/city_info.json"

    ############### Loading image data from data folder ######
    
    # Confirming that we are using Pittsburgh data 
    cityName = utils.readJsonFile(cityConfigFilePath)['city_name']
    if(cityName!= "PIT"):
        print ("Data does not belong to Pittsburgh")
        return -1 
    else:
        print ("We have Pittsburgh data")

    # Varying the data loading and calibration depending on the mode 
    if sfmMode == "stereo":
        print ("SFM in stereo mode")

        # Path to the stereo front left and front right folders
        stereoFrFolder = dataFolderPath + "/stereo_front_right"
        stereoFlFolder = dataFolderPath + "/stereo_front_left" 
        # Paths to all the images in respective folders 
        stereoFrImgPaths = utils.extractImagePaths(stereoFrFolder)
        stereoFlImgPaths = utils.extractImagePaths(stereoFlFolder)

        #Calib dictionary for right and left camera      
        stereoFrCalib= utils.readJsonFile(calibConfFilePath)['camera_data_'][7]
        stereoFlCalib= utils.readJsonFile(calibConfFilePath)['camera_data_'][6]

        print (stereoFrCalib)
        print (stereoFlCalib)

        input("Press Enter to continue")
        #stereoFlCalib= camData['image_raw_stereo_front_right']


    #TO DO: Loop through all images and perform exhaustive matching and add all 3D information to a global list 
  
    imgR = cvUtils.readImage(stereoFrImgPaths[0])
    imgL = cvUtils.readImage(stereoFlImgPaths[0])
    
    # Extract matches from the first two images from the stereo folder 
    [pts1, pts2, matches]= cvUtils.featureMatching(imgR, imgL, descriptorType, topNMatches, displayMatches=True, displayKp=False)

    # Create a vector of ones 
    onesVec= np.ones(len(matches))
    onesVec= np.reshape(onesVec, (onesVec.shape[0],1))

    # Create a set of homogenous points 
    pts1Homo= np.hstack((pts1, onesVec))
    pts2Homo= np.hstack((pts2, onesVec))

    # Compute camera projection matrices
    # To Do: This code needs K1, K2, R,t from the 
    M1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    M2 = np.hstack((R, t))
    C1  = K1 @ M1
    C2 =  K2 @ M2
    # 3D point in homogenous form ([x,y,z,w])
    point3dHomo = cv2.triangulatePoints(C1, C2, pts1.T, pts2.T).T
    # 3D point after normalizing by w aka ([x/w, y/w, z/w, 1])
    point3d = point3dHomo / point3dHomo[:, -1].reshape(-1,1)


    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_aspect('equal')

    X = point_3d[:,0]
    Y = point_3d[:,1] 
    Z = point_3d[:,2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # For each set of style and range settings, plot n random points in the box

    ax.scatter(X, Y, Z, s= 2)

    '''
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    '''

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


  
if __name__== "__main__":
  main()

