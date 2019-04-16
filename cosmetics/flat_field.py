#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 17:12:26 2019

Tools for making a flat field

@author: cheetham
"""

from astropy.io import fits
import numpy as np

try:
    import bottleneck
    nanmedian = bottleneck.nanmedian
    nanmean = bottleneck.nanmean
except:
    nanmedian = np.nanmedian
    nanmean = np.nanmean

#import matplotlib.pyplot as plt
#import time

def make_sphere_irdis_flat(flat_files,dark_files,save_name = None):
    '''
    Make a flat field using SPHERE-IRDIS flat files
    '''
    
    data = []
    dits = []
    output_header = fits.Header()
    
    for ix,fname in enumerate(flat_files):
        cube,header = fits.getdata(fname,header=True)
        
        cube = nanmean(cube,axis=0)
        dit = header['ESO DET SEQ1 REALDIT']
        
        data.append(cube)
        dits.append(dit)
        if ix == 0:
            output_header=header
        
        # Add some info to the header
        output_header['HIERARCH SAMPY FLAT FILE '+str(ix)] = fname[-40:] # take last
        output_header['HIERARCH SAMPY FLAT DIT '+str(ix)] = dit
        
    data = np.array(data)
    dits = np.array(dits)
    
    darks = []
    dark_dits = []
    for ix,fname in enumerate(dark_files):
        cube,header = fits.getdata(fname,header=True)
        cube = nanmean(cube,axis=0)
        darks.append(cube)
        dark_dits.append(header['ESO DET SEQ1 REALDIT'])
        
        output_header['HIERARCH SAMPY FLAT DARK '+str(ix)] = fname[-40:]
        output_header['HIERARCH SAMPY FLAT DARK DIT '+str(ix)] = dit
        
    darks = np.array(darks)
    dark_dits = np.array(dark_dits)
    
    # Make a master dark frame by dividing each frame by its exposure time
    # and then averaging        
    master_dark = darks/dark_dits[:,np.newaxis,np.newaxis]
    master_dark = nanmedian(master_dark,axis=0)
    
    # Now calculate the flat by subtracting the predicted dark from each frame
    # and then averaging
    darks = np.repeat(master_dark[np.newaxis,:,:],data.shape[0],axis=0)
    darks = darks*dits[:,np.newaxis,np.newaxis]
    
    master_flat = data - darks
    master_flat /= dits[:,np.newaxis,np.newaxis]
    master_flat = nanmedian(master_flat,axis=0)
    
    master_flat /= nanmedian(master_flat)

    # Diagnostic plots
#    plt.figure(1)
#    plt.clf()
#    plt.hist(master_flat.ravel(),bins=1000)
#    plt.figure(2)
#    plt.clf()
#    plt.imshow(master_flat)
#    plt.colorbar()
    
    if save_name:
        print('Saving flat as: '+save_name)
        fits.writeto(save_name,master_flat,output_header,overwrite=True,
                     output_verify = 'silentfix')
    
    return master_flat
