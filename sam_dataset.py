#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:36:39 2019

@author: cheetham
"""

import numpy as np
from astropy.io import fits
from instruments import sphere
from instruments.info import new_info
import matplotlib.pyplot as plt
import sys
import pickle

# Use FFTW if possible
try:
    import pyfftw.interfaces.numpy_fft as fft
except:
    from numpy import fft
    

class SAM_dataset(object):
    ''' Base object for dealing with sparse aperture masking datasets
    
    '''
    
    def __init__(self,files,headers=None,skies=None,flatfield=None,bad_pixels=None):
        ''' 
        
        files: list of files containing SAM data. It's assumed that the data is 
                in the first extension.
                Can also be a 3D or 4D array of SAM frames.
        skies: list of sky files
        flatfield: 2D or 3D array
        bad_pixels: 2D or 3D array
        
        Output attributes:
            data = 4D array (wavelength, frame, X, Y)
        
        '''

        data = []
        headers = []
        
        # If only 1 file is input, put it into an array to be processed the same way
        if isinstance(files,str):
            files = [files]
        
        if isinstance(files,list):
            
            # Set up the array to read the header information
            info = new_info()
            
            # Loop through the files and load them
            for cube_ix,f in enumerate(files):
                cube,hdr = fits.getdata(f,header=True)
                
                # Instrument specific steps.
                instrument = hdr['INSTRUME']
                ### Add your own instrument here ##
                if instrument == 'SPHERE':
                    cube,info = sphere.preproc_sphere(cube,hdr,info,cube_ix)
                elif instrument == 'NACO':
                    # TODO: Add NACO support
                    pass
                elif instrument == 'GPI':
                    # TODO: Add GPI support
                    pass
                
                
                data.append(cube)
                headers.append(hdr)
        elif isinstance(files,np.ndarray):
            data = files
            headers = headers
        else:
            print('Unrecognised file type for files')
            raise IOError()
        
        self.data = data
        self.headers = headers
        self.info = info
    
    def save(self,save_name):
        ''' Save all the info to a file so we don't have to repeat 
        calculations'''
        with open(save_name,'w') as myf:
            pickle.dump(self,myf)
    
    def get_cvis_ft(self,ft,mask,method='interpolate',wav_ix=0):
        """ Measure the complex visibilities from a Fourier transform of
        a 2D image.
        Needs a sampy.mask.Mask object as input
        wav_ix: The index of the wavelength used (if the image is 2D but 
                                                  the mask is for an IFS)
        """
        # Separate amplitudes and phases
        ft_ph = np.angle(ft)
        ft_amp = np.abs(ft)
        
        bl_pix = mask.bl_pix  # pixels that each baseline corresponds to
        if bl_pix.ndim == 3:
            bl_pix = mask.bl_pix[wav_ix]

        if method == 'interpolate':
            from scipy.interpolate import RectBivariateSpline
            
            # set up coordinate grid and interpolator
            x = np.arange(mask.npix) - mask.npix/2
            ph_interp_func = RectBivariateSpline(x,x,ft_ph)
            amp_interp_func = RectBivariateSpline(x,x,ft_amp)
            
            recon_amps = amp_interp_func(bl_pix[:,1],bl_pix[:,0],grid=False)
            recon_phs = ph_interp_func(bl_pix[:,1],bl_pix[:,0],grid=False)
            cvis = recon_amps*(np.cos(recon_phs)+np.complex(0,1)*np.sin(recon_phs))
            
        elif method == 'nearest':
            # Take the nearest pixel for each baseline
            bl_pix_rounded = np.round(bl_pix,decimals=0).astype(np.int)
            pix_in_image = bl_pix_rounded + [ft.shape[0]/2,ft.shape[1]/2]
            cvis = ft[pix_in_image[:,1],pix_in_image[:,0]]
            
        else:
            raise Exception('Unknown method in get_cvis_ft!')
        return cvis
    
    
    def measure_observables_from_cube(self,cube,mask,method='interpolate',
                                      plot=True,plot_title=''):
        '''Measure the complex visibility, closure phase, bispectral amplitude
        and square visibility from each image in a cube and save it all into arrays'''
        
        if cube.ndim ==4:
            n_frames = cube.shape[1]
            n_wav = cube.shape[0]
        else:
            n_frames = cube.shape[0]
            n_wav = 0
            
        # Do all the Fourier transforms now
        temp = fft.fftshift(cube,axes=(-2,-1))
        fts = fft.fftshift(fft.fft2(temp),axes=(-2,-1))

        # Set up the arrays to hold the observables
        cvis_all = np.zeros((n_wav,n_frames,mask.n_bl),dtype=np.complex)
#        v2_all = np.zeros((n_wav,n_frames,mask.n_bl))
        bs_all = np.zeros((n_wav,n_frames,mask.n_bs),dtype=np.complex)
#        cp_all = np.zeros((n_wav,n_frames,mask.n_bs))
        
        for wav_ix in range(n_wav):
            if cube.ndim == 4:
                wav_fts = fts[wav_ix]
            else:
                wav_fts = fts
                        
            for frame_ix in range(n_frames):
                
                ft = wav_fts[frame_ix]
                
                # Normalise by peak flux
                ft /= np.max(np.abs(ft))
                
                # Get the complex visibility first
                cvis_all[wav_ix,frame_ix] = self.get_cvis_ft(ft,mask,method=method)
        
                # Convert to bispectrum now
                bs_all[wav_ix,frame_ix] = mask.cvis_to_bispectrum(cvis_all[wav_ix,frame_ix])

        # Convert to v2 and closure phase
        v2_all = np.real(np.abs(cvis_all)**2)
        cp_all = np.angle(bs_all)
        
        
#        cp = np.nanmean(cp_all,axis=1)
#        cp_err = np.nanstd(cp_all,axis=1)
#        
        v2 = np.nanmean(v2_all,axis=1)
        v2_err = np.nanstd(v2_all,axis=1)
#        
#        cvis = np.nanmean(cvis_all,axis=1)
#        cvis_err = np.nanstd(cvis_all,axis=1)
        
        # Plot them
        if plot:
            plt.figure('Closure Phase')
            plt.clf()
            plt.hist(cp_all.ravel()*180./np.pi,bins=np.int(np.sqrt(cp_all.size)))
            plt.xlabel('Raw Closure phase (deg)')
            plt.ylabel('Number')
            plt.title(plot_title)
            
            plt.figure('V2')
            plt.clf()
            bl_length = np.sqrt(np.sum(mask.bl_coords**2,axis=1))
            for wav_ix in range(n_wav):
                plt.plot(bl_length,v2_all[wav_ix].T,'k.')
                plt.errorbar(bl_length,v2[wav_ix],yerr=v2_err[wav_ix],fmt='ro',zorder=5)
            plt.xlabel('Baseline length (m)')
            plt.ylabel('Raw V2')
            plt.ylim(ymin=0)
            plt.title(plot_title)

            plt.pause(0.01)
        
        return [cvis_all,bs_all,v2_all,cp_all]
    
    def measure_observables_from_sequence(self,mask,method='interpolate',
                                      plot=True,window_fwhm=None):
        '''Wrapper function around measure_observables_from_cube
        and get_cvis_ft that operates on an entire observing sequence
        to measure the complex visibility, V2, bispectrum and closure phase'''

        n_cubes = len(self.data)
        n_wav = self.data[0].shape[0]
        
        # Set up arrays for loop
        cvis_all = []
        v2_all = []
        bs_all = []
        cp_all = []
        
        # And average / uncertainty quantitiesanpt
        cvis = np.zeros((n_cubes,n_wav,mask.n_bl),dtype=np.complex)
        v2 = np.zeros((n_cubes,n_wav,mask.n_bl))
        bs = np.zeros((n_cubes,n_wav,mask.n_bs),dtype=np.complex)
        cp = np.zeros((n_cubes,n_wav,mask.n_bs))

        cvis_err = np.zeros((n_cubes,n_wav,mask.n_bl),dtype=np.complex)
        v2_err = np.zeros((n_cubes,n_wav,mask.n_bl))
        bs_err = np.zeros((n_cubes,n_wav,mask.n_bs),dtype=np.complex)
        cp_err = np.zeros((n_cubes,n_wav,mask.n_bs))
        
        for cube_ix,cube in enumerate(self.data):
#            print('Up to cube '+str(cube_ix)+' of '+str(n_cubes)+' \r')
            sys.stdout.write("Up to cube "+str(cube_ix)+" of "+str(n_cubes)+" \n\r")
            sys.stdout.flush()
            
            if window_fwhm:
                # Apply a super-Gaussian window to reduce the influence of readnoise
                x,y=np.indices(cube.shape[-2:])
                x -= cube.shape[-2]/2
                y -= cube.shape[-1]/2
                dist = np.sqrt(x**2+y**2)
                # The magic number below ensures window_fwhm is the fwhm in pixels
                window = np.exp(-(dist/(window_fwhm/2)*0.91244)**4)
                window /= np.max(window)
                cube *= window
            
            temp = self.measure_observables_from_cube(cube,mask,method=method,
                                  plot=plot,plot_title='Cube '+str(cube_ix))
            [cvis_cube,bs_cube,v2_cube,cp_cube] = temp
            
            cvis_all.append(cvis_cube)
            v2_all.append(v2_cube)
            bs_all.append(bs_cube)
            cp_all.append(cp_cube)
            
            # Calculate the mean and uncertainty across the cube
            cvis[cube_ix] = np.nanmean(cvis_cube,axis=1)
            v2[cube_ix] = np.nanmean(v2_cube,axis=1)
            bs[cube_ix] = np.nanmean(bs_cube,axis=1)
            cp[cube_ix] = np.nanmean(cp_cube,axis=1)
            
            # Uncertainty via SEM
            # It would be good to use the actual number of finite values for
            # each array instead of just sqrt(n_frames), but that would
            # mean breaking the vectorisation here
            n_frames = cvis_cube.shape[1]
            cvis_err[cube_ix] = np.nanstd(cvis_cube,axis=1) / np.sqrt(n_frames)
            v2_err[cube_ix] = np.nanstd(v2_cube,axis=1) / np.sqrt(n_frames)
            bs_err[cube_ix] = np.nanstd(bs_cube,axis=1) / np.sqrt(n_frames)
            cp_err[cube_ix] = np.nanstd(cp_cube,axis=1) / np.sqrt(n_frames)

        
        self.cvis_all = cvis_all
        self.v2_all = v2_all
        self.bs_all = bs_all
        self.cp_all = cp_all
        
        self.cp = cp
        self.cp_err = cp_err
        self.cvis = cvis
        self.cvis_err = cvis_err
        self.v2 = v2
        self.v2_err = v2_err
        self.bs = bs
        self.bs_err = bs_err
