# -*- coding: utf-8 -*-
"""
author: F.G. Heslinga
fgheslinga@gmail.com

"""
# Import libraries
import sys
import os
import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image

import numba
from numba import cfunc, carray
from numba.types import intc, CPointer, float64, intp, voidptr
from scipy import LowLevelCallable

""" Parameters and locations """
input_size  = (3744,3744)  # Resize sizes
output_size = (512,512)
filter_size = 21
save_type   = '.tiff'
open_type   = '.jpg'
open_dir    = 'E:/T2D-from-fundus/data/images_by_date/'
save_dir    = 'E:/T2D-from-fundus/data/norm_512/'
filenames_dir = 'E:/T2D-from-fundus/data/T2D_features_combined_20210802.xlsx'

""" Get filenames """
dtype_dic={'Fundus_Path': str}
data = pd.read_excel(filenames_dir, engine='openpyxl', dtype=str)
file_dirs  = data['orig_path_dir']
filenames  = data['Filename']

#filenames_train = data.loc[data['TrainValTest'] == 'Training']['Filename'].astype(str).values.tolist()
#filenames_val   = data.loc[data['TrainValTest'] == 'Validation']['Filename'].astype(str).values.tolist()
#filenames_test  = data.loc[data['TrainValTest'] == 'Test']['Filename'].astype(str).values.tolist()

""" Use Numba JIT to speed up filtering """
def jit_filter_function(filter_function):
    jitted_function = numba.jit(filter_function, nopython=True)
    @cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
    def wrapped(values_ptr, len_values, result, data):
        values = carray(values_ptr, (len_values,), dtype=float64)
        result[0] = jitted_function(values)
        return 1

    sig = None
    if sys.platform == "win32":
        sig = "int (double *, npy_intp, double *, void *)"
    return LowLevelCallable(wrapped.ctypes, signature=sig)

@jit_filter_function
def fnanstd(values):
    return np.nanstd(values)

@jit_filter_function
def fnanmean(values):
    return np.nanmean(values)

""" Define filtering """
def filter_im(im):
    # split into channels
    red   = im[:,:,0]
    green = im[:,:,1]
    blue  = im[:,:,2]
    
    # use ndiamge.generic_filter in combination with Numba JIT to obtain mean & std maps
    red_mean   = ndimage.generic_filter(red, function = fnanmean, size=(filter_size, filter_size))#, mode='nearest')
    green_mean = ndimage.generic_filter(green, function = fnanmean, size=(filter_size, filter_size))#, mode='nearest')
    blue_mean  = ndimage.generic_filter(blue, function = fnanmean, size=(filter_size, filter_size))#, mode='nearest')
    
    red_std   = ndimage.generic_filter(red, function = fnanstd, size=(filter_size, filter_size))#, mode='nearest')
    green_std = ndimage.generic_filter(green, function = fnanstd, size=(filter_size, filter_size))#, mode='nearest')
    blue_std  = ndimage.generic_filter(blue, function = fnanstd, size=(filter_size, filter_size))#, mode='nearest')

    red_filtered   = np.clip(((red - red_mean)     / (red_std + 20)   * 255 + 128),0,255).astype(np.uint8)
    green_filtered = np.clip(((green - green_mean) / (green_std + 20) * 255 + 128),0,255).astype(np.uint8)
    blue_filtered  = np.clip(((blue - blue_mean)   / (blue_std + 20)  * 255 + 128),0,255).astype(np.uint8)
    filtered_image = np.dstack((red_filtered, green_filtered, blue_filtered))   
    return filtered_image

""" Loop over all filenames """
def filter_from_orig_location(filenames, open_dir, save_dir, open_type, save_type):
    clean_filenames2 = []
    tot_num = len(filenames)
    for num_im in range(tot_num):
        if num_im % 100 == 0:
            print(num_im, " of " , tot_num)
        try:
            open_path =  os.path.join(open_dir, file_dirs[num_im]) + open_type
            save_path = os.path.join(save_dir, filenames[num_im]) + save_type
            im        = Image.open(open_path).crop((120,120,3625,3625)).resize(output_size,Image.BICUBIC)#open, remove black space, and resize
            im_array  = np.array(im).astype(float)
            im_masked = np.where(im_array > 10, im_array, np.NaN) #replace background with NaN for nanmean
            filt_im   = filter_im(im_masked)
            im_filtered = Image.fromarray(filt_im)
            im_filtered.save(save_path)
            clean_filenames2.append(filenames[num_im])
        except:
            pass
    return clean_filenames2
    
"""Apply filtering """
clean_filenames2 = filter_from_orig_location(filenames, open_dir, save_dir, open_type, save_type)
 
           
#filter_from_orig_location(filenames_train, open_dir, save_dir_train, open_type, save_type)
#filter_from_orig_location(filenames_val,   open_dir, save_dir_val,   open_type, save_type)
#filter_from_orig_location(filenames_test,  open_dir, save_dir_test,  open_type, save_type)
#
#""" Loop over all directories """
#def filter_from_orig_location(filenames, open_dir, save_dir, open_type, save_type):
#    tot_num_train = len(filenames)
#    num_im = 0
#    for datefolder in os.listdir(open_dir):
#        date = os.path.join(open_dir + datefolder)
#        cases = [f.name for f in os.scandir(date) if f.is_dir()]
#        for case in cases:
#            if case in filenames:
#                num_im+=1
#                if num_im % 100 == 0:
#                    print(num_im , " of " , tot_num_train)
#                open_path = os.path.join(date + "/" + case + "/" + case + open_type)
#                save_path = os.path.join(save_dir, case + save_type)
#                im        = Image.open(open_path).crop((120,120,3625,3625)).resize(output_size,Image.BICUBIC)#open, remove black space, and resize
#                im_array  = np.array(im).astype(float)
#                im_masked = np.where(im_array > 10, im_array, np.NaN) #replace background with NaN for nanmean
#                filt_im   = filter_im(im_masked)
#                im_filtered = Image.fromarray(filt_im)
#                im_filtered.save(save_path)  
#                
