#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 17:08:39 2019

A collection of bad pixel functions

@author: cheetham
"""
import numpy as np
from scipy import ndimage
from astropy.io import fits

def bad_pix_flat_detect(flat_input,n_sigma=7,min_cutoff=0.5,max_cutoff=1.5):
    """ Detect bad pixels from outliers in the flat field
    flat_input can be a file string or a numpy array
    n_sigma: Reject any pixels more than n_sigma from the median
    min_cutoff: Reject any pixels less than min_cutoff
    max_cutoff: Reject any pixels greater than max_cutoff
    """
    
    if isinstance(flat_input,str):
        flat = fits.getdata(flat_input)
    elif isinstance(flat_input,np.ndarray):
        flat = flat_input
    
    med = np.nanmedian(flat)
    mad = np.nanmedian(np.abs(flat-med))
    
    hot_pix = (flat - med) > (n_sigma*mad*1.4826) # number converts MAD to St.Dev.
    cold_pix = (flat - med) < (-n_sigma*mad*1.4826) # number converts MAD to St.Dev.
    
    # Apply the cutoffs in response
    hot_pix += (flat > max_cutoff)
    cold_pix += (flat < min_cutoff)
    
    bad_pix = hot_pix + cold_pix
    
    print('Found: '+str(bad_pix.sum())+' bad pixels from flat')
    
    return bad_pix
    

def cosmic_ray_detect(image,box_radius=3,n_sigma=7,silent=False):
    """ Cosmic ray detection function.
    Works by comparing each pixel to the local median absolute deviation.
    If the pixel value is more than n_sigma * MAD from the local median, it is called bad
    """

    box_radius = np.int(box_radius)
    box_size = np.int(2*box_radius + 1)  #make sure width is odd so the target pixel is centred

    # Get the median value in a box around each pixel (to use to calculate the MAD)
    median_vals = ndimage.median_filter(image,size=box_size)

    # Rather than loop through pixels, loop through the shifts and calculate 
    #  the deviation for all pixels in the image at the same time
    n_stdev_vals = box_size**2 -1 # We will ignore the centre pixel
    stdev_array = np.zeros((image.shape[0],image.shape[1],n_stdev_vals))
    shift_index = 0
    for yshift in np.arange(-box_radius,box_radius+1):
        for xshift in np.arange(-box_radius,box_radius+1):

            # Don't include the pixel in the MAD calculation
            if xshift ==0 and yshift == 0:
                continue

            shifted_image = np.roll(image,(yshift,xshift),axis=(0,1))
            stdev_array[:,:,shift_index] = (shifted_image - median_vals)

            shift_index += 1

    med_abs_dev = np.nanmedian(np.abs(stdev_array),axis=2)
    n_sig_array = (image-median_vals) / (med_abs_dev*1.4826) # this number is to convert MAD to std. deviation

    bad_array = np.abs(n_sig_array) > n_sigma    

    # In case we want to check the bad pixels that we detected:
    # pyfits.writeto('cosmic_ray_array.fits',np.abs(n_sig_array),overwrite=True)

#    cosmic_rays = np.where(bad_array)
    n_bad = np.sum(bad_array)
    if not silent:
        print('  '+str(n_bad)+' cosmic rays detected using n_sigma='+str(n_sigma))

    return bad_array

def replace_by_mean(cube,bad_pix,box_radius=1):
    ''' Replaces the pixels in "cube" marked by "bad_pix" by the local
    mean in a box of size (2*box_radius+1) on each side.
    '''
    
    # Work on cubes by default
    if cube.ndim == 4:
        iter_cube = cube
    elif cube.ndim == 3:
        iter_cube = cube[np.newaxis,:,:]
    elif cube.ndim == 2:
        iter_cube = cube[np.newaxis,np.newaxis,:,:]
            
    for wav_ix in range(iter_cube.shape[0]):
        
        for frame_ix in range(iter_cube.shape[1]):
            
            frame = iter_cube[wav_ix,frame_ix]
            
            # Work on a list of the bad pixels
            # If bad_pix is a 3D array, assume the 1st dimension is wavelength
            if bad_pix.ndim == 4:
                bpix_list = np.where(bad_pix[wav_ix,frame_ix])
                frame[bad_pix[wav_ix,frame_ix]] = np.nan          
            elif bad_pix.ndim == 3:
                bpix_list = np.where(bad_pix[wav_ix])
                frame[bad_pix[wav_ix]] = np.nan
            else:
                bpix_list = np.where(bad_pix)
                frame[bad_pix] = np.nan
            
            for bpix_ix in range(bpix_list[0].size):
                
                xpix = bpix_list[0][bpix_ix]
                ypix = bpix_list[1][bpix_ix]
                
                xmin = np.max([0,xpix-box_radius])
                xmax = np.min([xpix+box_radius,iter_cube.shape[2]])
                ymin = np.max([0,ypix-box_radius])
                ymax = np.min([ypix+box_radius,iter_cube.shape[3]])
                
                val = np.nanmean(frame[xmin:xmax,ymin:ymax])
                iter_cube[wav_ix,frame_ix,xpix,ypix] = val
            
    if cube.ndim == 4:
        cube = iter_cube
    elif cube.ndim == 3:
        cube = iter_cube[0]
    elif cube.ndim == 2:
        cube = iter_cube[0,0]
    
    return cube

def replace_by_const(cube,bad_pix,const=np.nan):
    ''' Replaces the pixels in "cube" marked by "bad_pix" by a constant
    (or NaN). Useful if we want to mask them out. This is just a wrapper
    '''
    
    # Work on cubes by default
    if cube.ndim == 4:
        iter_cube = 1*cube
    elif cube.ndim == 3:
        iter_cube = 1*cube[np.newaxis,:,:]
    elif cube.ndim == 2:
        iter_cube = 1*cube[np.newaxis,np.newaxis,:,:]
            
    for wav_ix in range(iter_cube.shape[0]):
        
        for frame_ix in range(iter_cube.shape[1]):
            
            frame = iter_cube[wav_ix,frame_ix]
            frame[bad_pix[wav_ix]] = const
            iter_cube[wav_ix,frame_ix] = frame
            
    if cube.ndim == 4:
        iter_cube = iter_cube
    elif cube.ndim == 3:
        iter_cube = iter_cube[0]
    elif cube.ndim == 2:
        iter_cube = iter_cube[0,0]
    
    return iter_cube
