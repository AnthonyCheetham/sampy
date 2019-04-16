#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:45:53 2019

@author: cheetham
"""

import sampy
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle
import pandas as pd
import time

# Test 1 : SPHERE-IRDIS
base_dir = '/Users/cheetham/data/sphere_data/HD142527_2017_SAM/IRDIS/'
cal_dir = base_dir+'Cal/'
data_dir = base_dir + 'Raw/'
save_dir = base_dir+'sampy/'

# Test 2: SPHERE-IFS
#base_dir = '/Users/cheetham/data/sphere_data/HD100546_SAM/IFS/'
#data_dir = base_dir + 'Cubed/'
#save_dir = base_dir+'sampy/'

#save_dir = base_dir + 'Analysis/'

test_flat = False
test_badpix = False
test_centring = False
test_quick_irdis_clean = False

make_mask = True
test_clps = False

test_full_calculation= False
test_saving = False
compare_with_idl = True

crop_size = 256

files = glob.glob(data_dir+'*OBS*.fits')
files.sort()

save_name = save_dir+'sam_data.pick'
save_name2 = save_dir+'sam_data.hdf'
    
#########

flat_outfile = base_dir+'flat.fits'
if test_flat:
    # Make a flat field
    flat_files = glob.glob(cal_dir+'*FLAT*.fits')
    dark_files = glob.glob(cal_dir+'*BCKG*.fits')
    flat_files.sort()
    dark_files.sort()
    sampy.cosmetics.flat_field.make_sphere_irdis_flat(flat_files,dark_files,save_name = flat_outfile)

if test_badpix:
    # Test flat bad pix
    bad_pix1 = sampy.cosmetics.bad_pixels.bad_pix_flat_detect(flat_outfile,n_sigma=2.5)
    cube1,hdr1 = fits.getdata(files[0],header=True)
    
    flat = fits.getdata(flat_outfile)
    
    for ix,im1 in enumerate(cube1):
        # Fix them
        im2 = im1 / flat
        im2 = sampy.cosmetics.bad_pixels.replace_by_mean(im1,bad_pix1,box_radius=1)
        
        
        bad_pix2 = sampy.cosmetics.bad_pixels.cosmic_ray_detect(im2,n_sigma=7)
        im3 = sampy.cosmetics.bad_pixels.replace_by_mean(im2,bad_pix2,box_radius=1)
        cube1[ix] = im3
    
    plt.clf()
    plt.imshow(im3,origin='lowerleft')
#    plt.imshow(bad_pix2)
    plt.colorbar()
    
    fits.writeto(files[0].replace(data_dir,save_dir),cube1,header=hdr1,
                 output_verify='silentfix',overwrite=True)

if test_centring:
    clean_files = glob.glob(save_dir+'*OBS*.fits')
    sam_data = sampy.sam_dataset.SAM_dataset(clean_files)
    
    mean_frame = np.nanmean(sam_data.data[0][0],axis=0)
    
    centre = sampy.cosmetics.recentre.find_centre(mean_frame,smooth_size=3)
    
    cen_frame = sampy.cosmetics.recentre.centre_on_pixel(mean_frame,centre)
    
    # Crop it
#    crop_frame = sampy.cosmetics.recentre.crop_array(cen_frame,256)
    
    
    # fft
#    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(crop_frame)))
#    pspec = np.real(ft*np.conj(ft))
#    ph = np.angle(ft)
    
    plt.figure(2)
    plt.clf()
    plt.imshow(mean_frame,vmax=1000,vmin=-1000)
    plt.colorbar()
    plt.figure(1)
#    plt.imshow(crop_frame,origin='lowerleft')
#    plt.imshow(ph,origin='lowerleft')
#    plt.imshow(np.sqrt(pspec),origin='lowerleft',vmax=1e4)
#    plt.imshow(mean_frame)
#    plt.colorbar()
    
if test_quick_irdis_clean:
    
    sam_data = sampy.sam_dataset.SAM_dataset(files)
    flat_data = sampy.sam_dataset.SAM_dataset(flat_outfile)
    flat = flat_data.data[0]
    
    for cube_ix,cube in enumerate(sam_data.data):
        
        if cube_ix < 40:
            continue
        
        print('Cube number: '+str(cube_ix)+' of '+str(len(sam_data.data)))
        
        # Flat field
        cube /= flat
        
        # Clean from bad pixels
        bad_pix_flat = sampy.cosmetics.bad_pixels.bad_pix_flat_detect(flat,n_sigma=2.5,
                                      min_cutoff=0.5,max_cutoff=1.5)
        bad_pix_flat = bad_pix_flat[:,0,:,:]

        # Mask the bad pixels for the centre detection.
        # we will fix them later when we've cropped most of them away
        masked_cube = sampy.cosmetics.bad_pixels.replace_by_const(cube,bad_pix_flat,const=np.nan)

        # Do a quick centring and crop before fixing the bad pixels to save time
        cen_cube = []
        cropped_badpix = []
        for wav_ix,wav_cube in enumerate(masked_cube):
            mean_frame = np.mean(wav_cube,axis=0)
            centre = sampy.cosmetics.recentre.find_centre(mean_frame,smooth_size=10)
        
            cen_wav_cube = sampy.cosmetics.recentre.centre_on_pixel(cube[wav_ix],centre)
            cen_wav_cube = sampy.cosmetics.recentre.crop_array(cen_wav_cube,crop_size)
            cen_cube.append(cen_wav_cube)
            cropped_badpix.append(sampy.cosmetics.recentre.crop_array(bad_pix_flat[wav_ix],crop_size))

        cen_cube = np.array(cen_cube)
        cropped_badpix = np.array(cropped_badpix)
        
        # Fix the bad pixels from the flat
        cen_cube = sampy.cosmetics.bad_pixels.replace_by_mean(cen_cube,cropped_badpix,box_radius=1)
        clean_cube = np.zeros(cen_cube.shape)
        for wav_ix, wav_cube in enumerate(cen_cube):
            # Find other static bad pixels from the median frame
            med_frame = np.nanmedian(wav_cube,axis=0)
            static_bad_pix = sampy.cosmetics.bad_pixels.cosmic_ray_detect(med_frame,n_sigma=5)
            # Correct these
            clean_cube[wav_ix] = sampy.cosmetics.bad_pixels.replace_by_mean(wav_cube,static_bad_pix,box_radius=2)
            
            # Now search for cosmic rays in the individual frames
            for frame_ix,frame in enumerate(clean_cube[wav_ix]):
                # Fix the bad pixels based on deviation from their neighbours
                bad_pix_cosmic = sampy.cosmetics.bad_pixels.cosmic_ray_detect(frame,n_sigma=7,silent=True)
                clean_cube[wav_ix,frame_ix] = sampy.cosmetics.bad_pixels.replace_by_mean(frame,bad_pix_cosmic,box_radius=1)

        # Save the results
        fits.writeto(files[cube_ix].replace(data_dir,save_dir+'clean_'),clean_cube,
                     header=sam_data.headers[cube_ix],output_verify='silentfix',overwrite=True)
        
        # Clean up arrays just in case python doesn't do it very well
        masked_cube = 0
        clean_cube = 0
        cen_cube = 0 
        cen_wav_cube = 0
        mean_frame = 0

if make_mask:
    # Make the mask
    rot_angle=0.0*np.pi/2 #rotation angle
    scale = 1.03
    npix = 256
    
    # Info that we need
    #in mm from centre of mask. Need in metres in pupil.
    xy_coords= np.array([[-1.894,1.894,-3.788,-1.894,-3.788,3.788,0],
                         [3.7179,3.7179,0.4374,-0.6561,-1.7496,-1.7496,-3.9366]] ).T
    mirrorsz=8.1
    #holesize=1.375mm
    masksz=10.5# No idea.
    wavel = [2100e-9,2251e-9] # metres
    pixel_scale = 12.267/1.0015 # mas/pixel
    pixel_scale_rad = pixel_scale * np.pi/(180*3600*1000)
    xy_coords*=scale*mirrorsz/masksz #also converts to metres

    mask = sampy.mask.Mask(xy_coords,pixel_scale,npix,wavel,'',
                           stretch_factor=1.,rotation_rad=rot_angle,
                           round_to_pixel=False)


if test_clps:
    clean_files = glob.glob(save_dir+'clean_*OBS*.fits')
    sam_data = sampy.sam_dataset.SAM_dataset(clean_files[4:5])
    
    wav_ix = 0
    obs_ix = 0
    cube = sam_data.data[obs_ix][wav_ix]
    data_vis = np.zeros((cube.shape[0],mask.n_bl),dtype=np.complex)
    cps = np.zeros((cube.shape[0],mask.n_bs))
    bs = np.zeros((cube.shape[0],mask.n_bs),dtype=np.complex)
    
    for fr_ix,frame in enumerate(cube):
        # HACK: shift the cube
        if fr_ix == 0:
            print('Shifting the cube to test the effect on clps')
        frame = np.roll(frame,0,axis=0)
    
        ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(frame)))
        ft_amp = np.abs(ft)
        ft_ph = np.angle(ft)
        
        ft /= np.max(ft_amp)
        ft_amp /= np.max(ft_amp)
        
        # Get complex visibilities
        cvis = sam_data.get_cvis_ft(ft,mask,method='interpolate')
        phases = np.angle(cvis)
        data_vis[fr_ix] = cvis
        # Turn to cps and bispectra
        cps[fr_ix] = np.dot(phases,mask.bl2bs)
        bs[fr_ix] = mask.cvis_to_bispectrum(cvis)
        # Make cps between -pi and pi
        cps[fr_ix] = ((cps[fr_ix]+np.pi) % (2*np.pi)) - np.pi
        
    plt.figure(2)
    plt.clf()
#    plt.imshow(ft_amp,origin='lowerleft',vmax=0.08,extent=[-(npix//2.)-0.5,-(-npix//2.)-0.5,
#                               -(npix//2.)-0.5,-(-npix//2.)-0.5])
    plt.imshow(ft_ph,origin='lowerleft',vmin=-0.5,vmax=0.5,extent=[-(npix//2.)-0.5,-(-npix//2.)-0.5,
                               -(npix//2.)-0.5,-(-npix//2.)-0.5])

    plt.plot(ft_coords[6,0],ft_coords[6,1],'rx')
    plt.plot(ft_coords[6,0],ft_coords[6,1],'rx')
    plt.xlim(-npix//4,npix//4)
    plt.ylim(-npix//4,npix//4)
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()

    
    cps = np.angle(bs)
    bl_coords = mask.bl_coords
    ft_coords = mask.bl_pix[wav_ix]
    npix = mask.npix
       
    plt.figure(1)
    plt.clf()
    plt.title('Mask')
    plt.plot(bl_coords[:,0],bl_coords[:,1],'x')
    plt.plot(-bl_coords[:,0],-bl_coords[:,1],'x')
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    
    
    plt.figure(3)
    plt.clf()
    lens = np.sqrt(bl_coords[:,0]**2+bl_coords[:,1]**2)
    for ix in range(mask.n_bl):
#        plt.plot(lens[ix],np.abs(data_vis[:,ix]),'x')
        plt.plot(np.repeat(lens[ix],cube.shape[0]),np.abs(data_vis[:,ix]),'x')
        
    # I made this plot to work out what the coordinates should be when
    # plotting the mask using "extent" instead of default coords.
    # But it turns out the default coords are related to "extent" using
    # the complicated relation below
#    plt.figure(4)
#    plt.clf()
#    npix = 4
#    fake_im = np.random.normal(size=(npix,npix))
#    plt.imshow(fake_im,origin='lowerleft',extent=[-(npix//2.)-0.5,-(-npix//2.)-0.5,
#                               -(npix//2.)-0.5,-(-npix//2.)-0.5])
#    plt.plot(0,0,'rx')
    
    plt.figure(5)
    plt.clf()
#    plt.hist(cps.ravel(),bins=10)
    plt.plot(cps.ravel()*180./np.pi,'x')
    plt.ylim(-5,5)

if test_full_calculation:
    
    clean_files = glob.glob(save_dir+'clean_*OBS*.fits')
    sam_data = sampy.sam_dataset.SAM_dataset(clean_files)

    import time
    t0 = time.time()
    sam_data.measure_observables_from_sequence(mask,method='interpolate',plot=True,window_fwhm=80)
    t1 = time.time()
    print('Time taken: '+str(t1-t0)+' secs')
#    sam_data.save(save_name)

if test_saving:
#    t0 = time.time()
#    with open(save_name,'r') as myf:
#        sam_data = pickle.load(myf)
#    t1 = time.time()
#    print t1-t0
    save = True
    load = True
    if save:
        t1 = time.time()
        n_data = len(sam_data.data)
        arr = []
        for ix in range(n_data):
            new_row = {'TargetName':str(ix),'cp':sam_data.cp[ix],'cp_err':sam_data.cp_err[ix],'cp_all':sam_data.cp_all[ix],
                       'cvis':sam_data.cvis[ix],'cvis_err':sam_data.cvis_err[ix],'cvis_all':sam_data.cvis_all[ix],
                       'bs':sam_data.bs[ix],'bs_err':sam_data.bs_err[ix],'bs_all':sam_data.bs_all[ix],
                       'v2':sam_data.v2[ix],'v2_err':sam_data.v2_err[ix],'v2_all':sam_data.v2_all[ix]}
            arr.append(new_row)
        df = pd.DataFrame(data=arr)
        df.to_hdf(save_name2,'sam_data',mode='w',format='fixed')
        t2 = time.time()
        print t2-t1
        
    if load:
        t1 = time.time()
        with pd.HDFStore(save_name2) as store:
            df = store['sam_data']
        t2 = time.time()
        print t2- t1

if compare_with_idl:
    with pd.HDFStore(save_name2) as store:
        df = store['sam_data']

    # Load the idl data
    from scipy.io.idl import readsav
    ix = 9
    bs_dict = readsav(base_dir+'Analysis/bs{0:04d}.idlvar'.format(ix))
    mf_file = readsav('/Users/cheetham/code/masking/templates/sphere/mf_7Hole_IRDIS_D_K12.idlvar')
    
    idl_cp_all = np.angle(bs_dict['bs_all'])
    idl_cp = bs_dict['cp']
    idl_cp_err = bs_dict['cp_sig']
    idl_u = bs_dict['u']
    
    sampy_cp = df['cp'][ix]
    sampy_cp_err = df['cp_err'][ix]
    sampy_cp_all = df['cp_all'][ix]
    
    x=np.arange(idl_cp.size)
    plt.figure(1)
    plt.clf()
#    plt.errorbar(x,idl_cp.ravel(),yerr=idl_cp_err.ravel(),fmt='x')
#    plt.errorbar(x,sampy_cp.ravel(),yerr=sampy_cp_err.ravel(),fmt='x')
    plt.errorbar(idl_cp.ravel(),sampy_cp.ravel(),xerr=idl_cp_err.ravel(),yerr=sampy_cp_err.ravel(),fmt='x')
    plt.ylim(-0.06,0.06)
    plt.xlim(-0.06,0.06)
    
    plt.figure(2)
    plt.clf()
    n_obs = sampy_cp_all.shape[-2]
    for ix in range(n_obs):
        plt.plot(x,idl_cp_all[:,:,ix].ravel(),'r.')
        plt.plot(x+0.2,sampy_cp_all[:,ix,:].ravel(),'b.')
    plt.plot(x,idl_cp.ravel(),'ko')
    plt.plot(x+0.2,sampy_cp.ravel(),'ko')
    
    
    