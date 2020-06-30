import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import glob
import os
import shutil
import re
import cv2
import imageio
import math

filepath = 'test/inference_000/*.*'
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def normalize(row):
    x,y,z = row
    sum = math.pow(x,2) + math.pow(y,2) + math.pow(z,2)
    sqrt_sum = math.sqrt(sum)
    
    x_norm = round(x / sqrt_sum , 3)
    y_norm = round(y / sqrt_sum  , 3)
    z_norm = round(-1 * z / sqrt_sum , 3)
    
    return x_norm, y_norm, z_norm 


def write_params():
    f = open('params', 'w')
    for filename in sorted(glob.glob(filepath), key=numericalSort):
        if '.npy' in filename and "parameters" in filename:
            print(filename)
            data = np.load(filename)
            data = np.round(data, 3)
            # data = normalize(data[0], data[1], data[2])
            # data = df.apply(normalize, axis=1)
            data = np.apply_along_axis(normalize, 1, data)
            print(str(data))
            f.write('{}\n-----------------------------\n'.format(filename))
            f.write('{}\n-----------------------------------------------------------\n'.format(data))

def separate_seg_images():
    for filename in glob.glob(filepath):
        if '.png' in filename and "segmentation" in filename and "final" in filename:
            print(filename)            
            shutil.copy(filename,"outputs/{}".format(filename))

def write_masks(filepath):
    f = open('masks','w')
    for filename in sorted(glob.glob(filepath), key= numericalSort):
        if '.npy' in filename and 'masks' in filename:
            print(filename)
            data = np.load(filename)
            print(data)

def apply_mask(mask_path,image_path, mask_index):
    image = cv2.imread(image_path)
    # mask = numpy.zeros_like(image)
    mask = np.load(mask_path)[mask_index]   
    # # copy your image_mask to all dimensions (i.e. colors) of your image
    # for i in range(3): 
    #     mask[:,:,i] = image.copy()

    # # apply the mask to your image
    # masked_image = image[mask]
    # cv2.imshow(masked_image)
    
    imageio.imwrite('mask.jpg', mask)


# separate_seg_images()
write_params()
# write_masks('test/inference/*.*')
# i=58
# apply_mask('inference/{}_plane_masks_0.npy'.format(i),'inference/{}_image_0.png'.format(i))
# apply_mask('/home/amr/Documents/Code/planercnn/test/inference_000/827_plane_masks_0.npy','/home/amr/Documents/Code/planercnn/example_images_3/827.png',5)
