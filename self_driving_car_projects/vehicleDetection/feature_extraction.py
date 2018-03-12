
import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog

def extract_hog_features(img, vis=True, pix_per_cell = 8, cell_per_block = 2, orient = 9, feature_vec=False):

    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                                  transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys', 
                       transform_sqrt=True, visualise=vis, feature_vector=feature_vec )
        return features    


def bin_spatial(img, color_space='RGB', size=(32, 32), unravel=True):
    # Convert image to new color space (if specified)
    # Use cv2.resize().ravel() to create the feature vector
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)             
    # Use cv2.resize().ravel() to create the feature vector
    if(unravel):
        features = cv2.resize(feature_image, size).ravel() 
        return features
    else: 
        return feature_image




def extract_color_hog_features(img,color_space='HLS', pix_per_cell = 8, cell_per_block = 2, orient = 5 ):

    color_image = bin_spatial(img, color_space, unravel=False)
    hog_features=[]
    for channel in range(color_image.shape[2]):
        hog_features.append(extract_hog_features(color_image[:,:,channel],vis=False,
                            orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, feature_vec=True))
    hog_features = np.ravel(hog_features)   
    return hog_features