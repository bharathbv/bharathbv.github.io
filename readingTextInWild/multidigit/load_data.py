

import os
from PIL import Image
import numpy as np
import pickle
import h5py as h5py
import matplotlib.pyplot as plt
import random

""" The DigitStructFile is just a wrapper around the h5py data.  It basically references """
#    inf:              The input h5 matlab file
#    digitStructName   The h5 ref to all the file names
#    digitStructBbox   The h5 ref to all struc data

class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

# getName returns the 'name' string for for the n(th) digitStruct. 
    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

# bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox. 
    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

# getBbox returns a dict of data for the n(th) bbox. 
    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox

    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s

# getAllDigitStructure returns all the digitStruct from the input file.     
    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

# Return a restructured version of the dataset (one structure by boxed digit).
#
#   Return a list of such dicts :
#      'filename' : filename of the samples
#      'boxes' : list of such dicts (one by digit) :
#          'label' : 1 to 9 corresponding digits. 10 for digit '0' in image.
#          'left', 'top' : position of bounding box
#          'width', 'height' : dimension of bounding box
#
# Note: We may turn this to a generator, if memory issues arise.
    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 'filename' : pictDat[i]["name"] }
            figures = []
            for j in range(len(pictDat[i]['height'])):
                figure = {}
                figure['height'] = pictDat[i]['height'][j]
                figure['label']  = pictDat[i]['label'][j]
                figure['left']   = pictDat[i]['left'][j]
                figure['top']    = pictDat[i]['top'][j]
                figure['width']  = pictDat[i]['width'][j]
                figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result



def display_rand_cropped_images(dataset,location):
    import os
    dataset_size = len(dataset)
    plt.rcParams['figure.figsize'] = (20.0, 20.0)
    x, ax = plt.subplots(nrows=1, ncols=10)
    y=-1
    for i in range(5):
        num = random.randrange(1, dataset_size)
        fin = os.path.join(location, dataset[num]['filename'])
        im = Image.open(fin)
    
        boxes = dataset[num]['boxes']
        if len(boxes) > 5:
            print(fin, "has more than 5 digits")
        else:
            left = [j['left'] for j in boxes]
            top = [j['top'] for j in boxes]
            height = [j['height'] for j in boxes]
            width = [j['width'] for j in boxes]
            lab = [j['label'] for j in boxes]
            im_left = min(left)
            im_top = min(top)
            im_height = max(top) + max(height) - im_top
            im_width = max(left) + max(width) - im_left

        im_top = im_top - im_height * 0.05 # a bit higher
        im_left = im_left - im_width * 0.05 # a bit wider
        im_bottom = np.amin([np.ceil(im_top + 1.2 * im_height), im.size[1]])
        im_right = np.amin([np.ceil(im_left + 1.2 * im_width), im.size[0]])
        box =[im_left, im_top, im_right, im_bottom]

        y= y+1
        ax[y].set_title('Actual-{}'.format(lab), loc='center')
        ax[y].imshow(im)
        size = (64,64)
        region = im.crop(box).resize(size, Image.ANTIALIAS)
        y= y+1
        ax[y].set_title('Cropped-{}'.format(lab), loc='center')
        ax[y].imshow(region)


def get_dataset(data, location):
    dataset = np.ndarray([len(data),64,64,3], dtype='float32')
    labels = np.ones([len(data), 6], dtype=int) * 10
    for i in range(len(data)): 
        fin = os.path.join(location, data[i]['filename'])
        im = Image.open(fin)
        boxes = data[i]['boxes']
        num_digits = len(boxes)
        if num_digits > 5:
            print(fin, "has more than 5 digits")
        else:
            left = [j['left'] for j in boxes]
            top = [j['top'] for j in boxes]
            height = [j['height'] for j in boxes]
            width = [j['width'] for j in boxes]
            lab = [j['label'] for j in boxes]
            labels[i,0] = num_digits
            for k in np.arange(num_digits):
                labels[i,k+1] = lab[k]

        im_left = min(left)
        im_top = min(top)
        im_height = max(top) + max(height) - im_top
        im_width = max(left) + max(width) - im_left

        im_top = im_top - im_height * 0.05 # a bit higher
        im_left = im_left - im_width * 0.05 # a bit wider
        im_bottom = np.amin([np.ceil(im_top + 1.2 * im_height), im.size[1]])
        im_right = np.amin([np.ceil(im_left + 1.2 * im_width), im.size[0]])
        box =[im_left, im_top, im_right, im_bottom]
        region = im.crop(box).resize([64,64], Image.ANTIALIAS)
        reg_array = np.array(region, dtype='float32')
        reg = np.dot(reg_array, [[0.2989],[0.5870],[0.1140]])
        mean = np.mean(reg, dtype='float32')
        std = np.std(reg, dtype='float32', ddof=1)
        if std < 1e-4: std = 1.
        reg = (reg - mean) / std
        dataset[i,:,:,:] = reg[:,:,:]
    return dataset, labels

def parse_digitStruct(train_location, extra_location, test_location):
    train = DigitStructFile(train_location)
    train_data = train.getAllDigitStructure_ByDigit()
    test = DigitStructFile(test_location)
    test_data = test.getAllDigitStructure_ByDigit()
    extra = DigitStructFile(extra_location)
    extra_data = extra.getAllDigitStructure_ByDigit()
    
    return train_data, extra_data, test_data

def display_random_images(dataset,location, num_images):

    dataset_size = len(dataset)
    plt.rcParams['figure.figsize'] = (20.0, 20.0)
    x, ax = plt.subplots(nrows=1, ncols=num_images)
    y = 0
    for i in range(num_images):
        num = random.randrange(1, dataset_size)
        fin = os.path.join(location, dataset[num]['filename'])
        im = Image.open(fin)
    
        boxes = dataset[num]['boxes']
        if len(boxes) > 5:
            print(fin, "has more than 5 digits")
        else:
            left = [j['left'] for j in boxes]
            top = [j['top'] for j in boxes]
            height = [j['height'] for j in boxes]
            width = [j['width'] for j in boxes]
            lab = [j['label'] for j in boxes]
            im_left = min(left)
            im_top = min(top)
            im_height = max(top) + max(height) - im_top
            im_width = max(left) + max(width) - im_left
        ax[y].set_title('{}'.format(lab), loc='center')
        ax[y].imshow(im)
        y += 1

def split_to_train_and_validation(combined_samples, combined_labels):
        total_num_samples = len(combined_samples) 
        num_train_samples =int(total_num_samples *0.8)
        num_validation_samples = total_num_samples - num_train_samples

        return combined_samples[0:num_train_samples,:,:,:], combined_labels[0:num_train_samples,:],combined_samples[num_train_samples:, :,:,:],combined_labels[num_train_samples:, :]
