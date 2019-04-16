#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:58:43 2019

@author: cheetham
"""
import numpy as np
import pdb

class Mask(object):
    """ Class defining the mask geometry. When launched, this will
    populate a range of fields that are useful to have
    """
    
    def __init__(self,xy_coords,pixel_scale_mas,npix,wavelength,
                 mask_file,stretch_factor=1.,rotation_rad=0.,
                 round_to_pixel=False):
        """
        xy_coords: n_holes x 2 array with coordinates of mask holes in metres
        pixel_scale_mas: angular size of one pixel in mas.
        npix: Number of pixels in the image (after cropping)
        wavelength: float or 1D array of wavelength values (for IFUs)
        
        # These are useful for making a mask match the data more closely
        stretch_factor: factor to stretch the xy coordinates
        rotation: angle to rotate the mask holes
        
        """

                
        n_holes = xy_coords.shape[0]
        n_bl = n_holes*(n_holes-1)/2
        n_bs = n_holes*(n_holes-1)*(n_holes-2)/6
        n_wav = np.size(wavelength)
        
        wavelength = np.atleast_1d(wavelength)
        
        pixel_scale_rad = pixel_scale_mas * np.pi/(180*3600*1000)

        self.xy_coords = xy_coords
        self.n_holes = n_holes
        self.n_bl = n_bl
        self.n_bs = n_bs
        self.pixel_scale_rad = pixel_scale_rad
        self.npix = npix
        
        ###### Apply mask stretching and rotation
        rot_xy_coords = np.zeros(xy_coords.shape)
        for ix,coords in enumerate(xy_coords):
           oldx,oldy = coords
           
           newx = oldx*np.cos(rotation_rad)-oldy*np.sin(rotation_rad)
           newy = oldy*np.cos(rotation_rad)+oldx*np.sin(rotation_rad)
           rot_xy_coords[ix,:] = [newx,newy]
    
        xy_coords = stretch_factor * rot_xy_coords
        
        # If we want to round all of the uv coordinates to the
        # nearest pixel, we need to shift xy_coords so it fits
        # onto a grid. This needs to be done at each wavelength!
        if round_to_pixel:
            onepix_amount = npix*pixel_scale_rad/wavelength[0]
            xy_coords = np.round(xy_coords*onepix_amount)/onepix_amount
            print('Coordinates of mask holes are being forced onto a regular grid!')

        
        ###### Start calculating Baseline quantities
                
        # Now calculate the baselines
        bl_coords = np.zeros((n_bl,2)) # in metres
        bl2h_ix = np.zeros((n_bl,2),dtype=int) # baseline to holes index
        h2bl_ix = np.zeros((n_holes,n_holes),dtype=int)-1
        # since there is no integer NaN in python, we have to use -1 to denote invalid entries above
        bl_ix = 0
        for h1 in range(n_holes-1):
            for h2 in np.arange(h1+1,n_holes):

                bl_coords[bl_ix] = xy_coords[h2] - xy_coords[h1]
                bl2h_ix[bl_ix] = h1,h2
                h2bl_ix[h1,h2] = int(bl_ix)
                
                bl_ix+=1

        bl_pix = np.zeros((n_wav,n_bl,2))        
        for wav_ix in range(n_wav):
            bl_pix_frac = pixel_scale_rad*(bl_coords/wavelength[wav_ix])
            bl_pix[wav_ix] = bl_pix_frac * npix # in pixels
        
        self.bl_coords = bl_coords # in metres
        self.bl_pix = bl_pix # in pixels
        self.bl2h_ix = bl2h_ix
        self.h2bl_ix = h2bl_ix
        
        ###### Start calculating Bispectral quantities
        
        bs_u = np.zeros((n_bs,3)) # u coords for each bispectrum triplet
        bs_v = np.zeros((n_bs,3)) # v coords for each bispectrum triplet
        bs2h_ix = np.zeros((n_bs,3),dtype=int) # bs2h_ix[x] = [hole1, hole2, hole3]
        bs2bl_ix = np.zeros((n_bs,3),dtype=int) # bs2bl_ix[x] = [baseline1, baseline2, baseline3]
        bl2bs = np.zeros((n_bl,n_bs),dtype=int) # Matrix to convert baseline measurements to bispectra
        
        # Loop through triplets of holes
        bs_ix = 0
        for h1 in range(n_holes-2):
            for h2 in np.arange(h1+1,n_holes-1):
                for h3 in np.arange(h2+1,n_holes):
                    # Fill in all the arrays to convert between quantities
                    bs2h_ix[bs_ix] = [h1,h2,h3]
                    bl1 = h2bl_ix[h1,h2]
                    bl2 = h2bl_ix[h2,h3]
                    bl3 = h2bl_ix[h1,h3] # note we actually mean the conjugate of this baseline
                    bs2bl_ix[bs_ix] = [bl1,bl2,bl3] 
                    
                    bs_u[bs_ix] = bl_coords[:,0][[bl1,bl2,bl3]]*[1,1,-1]
                    bs_v[bs_ix] = bl_coords[:,1][[bl1,bl2,bl3]]*[1,1,-1]
                    
                    bl2bs[bl1,bs_ix] = 1
                    bl2bs[bl2,bs_ix] = 1
                    bl2bs[bl3,bs_ix] = -1 # i.e. conjugate of this
                    
                    bs_ix += 1
        
        self.bs_u = bs_u
        self.bs_v = bs_v
        self.bs2h_ix = bs2h_ix
        self.bs2bl_ix = bs2bl_ix
        self.bl2bs = bl2bs

    def cvis_to_bispectrum(self,cvis):
        ''' Convert complex visibilities to bispectrum'''
        temp_cvis = cvis[self.bs2bl_ix]
        temp_cvis[:,-1] = np.conjugate(temp_cvis[:,-1])
        bispectrum = np.prod(temp_cvis,axis=1)
        return bispectrum
