#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 13:11:51 2019

@author: cheetham
"""

import numpy as np
import astropy.io.fits as fits
from scipy import signal
import pdb

import matplotlib.pyplot as plt


def find_centre(frame,smooth_size=5):
    """ Find the position of a SAM PSF in an image by smoothing 
    it and taking the peak
    smooth_size: FWHM of gaussian kernel used to smooth the image
    
    """
    
    # Remove nans first
    frame = np.nan_to_num(frame)
    
    # Smooth the image by convolution with a Gaussian
    x,y = np.indices((smooth_size*2+1,smooth_size*2+1)) # ensures an odd number of pix
    gauss = np.exp(- ((x-x.shape[0]/2)**2 + (y-y.shape[0]/2)**2)/(2*(smooth_size/2.355)**2))
    
    gauss /= np.sum(gauss)
    
    smoothed_frame = signal.fftconvolve(frame,gauss,mode='same')
    
    # Take the peak as the centre
    centre = np.unravel_index(np.argmax(smoothed_frame,axis=None),smoothed_frame.shape)
    
#    plt.clf()
#    plt.imshow(smoothed_frame,vmin=0)
#    plt.plot(centre[1],centre[0],'rx')
    
    return centre

def centre_on_pixel(frame,position):
    """ Shifts an image so it is centred on the pixel given in "position"
    Only integer positions are accepted
    """
    # Ensure it's an integer
    position = np.array(position).astype(int)
    
    shifts = np.array([frame.shape[-2]/2,frame.shape[-1]/2]) - position
    
    frame = np.roll(np.roll(frame,shifts[0],axis=-2),shifts[1],axis=-1)
    
    # Mask out the rows and columns that wrapped
    if shifts[0] >= 0:
        frame[...,0:shifts[0],:] = np.nan
    else:
        frame[...,shifts[0]:,:] = np.nan
        
    if shifts[1] >= 0:
        frame[...,:,0:shifts[1]] = np.nan
    else:
        frame[...,:,shifts[1]:] = np.nan
            
    return frame

def crop_array(input_array,size):
    """ Cut down or expand a frame so it is a given size
    """
    
    # In case size is an integer
    if np.size(size) ==1:
        size = [size,size]
    
    output_shape = list(input_array.shape)
    output_shape[-2:] = size
    output_array = np.zeros(output_shape,dtype=input_array.dtype)
    
    min_xsz = int(np.min([output_array.shape[-1], input_array.shape[-1]]))
    min_ysz = int(np.min([output_array.shape[-2], input_array.shape[-2]]))

    # Centred on the middle of each array, take a min_ysz x min_xsz region from one
    #  array and put it into the other
    # The ... means we apply this to only the last two dimensions
    output_array[..., output_array.shape[-2]//2-min_ysz//2:
        output_array.shape[-2]//2+min_ysz//2,
        output_array.shape[-1]//2-min_xsz//2:
        output_array.shape[-1]//2+min_xsz//2] = \
        input_array[...,input_array.shape[-2]//2-min_ysz//2:
        input_array.shape[-2]//2+min_ysz//2,
        input_array.shape[-1]//2-min_xsz//2:
        input_array.shape[-1]//2+min_xsz//2]
    
    return output_array

        