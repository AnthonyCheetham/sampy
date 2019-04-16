#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:53:18 2019

@author: cheetham
"""

import numpy as np
import external
import time

def preproc_sphere(cube,header,info,cube_ix):
    ''' Pre-Process SPHERE data.
    Gets the frames ready and in a consistent format with other instruments
    '''
    
    # Work out which module we're dealing with:
    module = header['ESO DET ID'] # IRDIS and IFS have it here
    if module == '':
        module = header['ESO DET DEV 1ID']
#    print('Camera: '+str(module))
        
    if module == 'IRDIS':
        cube, info = preproc_irdis(cube,header,info,cube_ix)
    elif module == 'IFS':
        cube, info = preproc_ifs(cube,header,info,cube_ix)
    elif module == 'ZIMPOL':
        cube, info = preproc_zimpol(cube,header,info,cube_ix)
        
    return cube,info
        

def preproc_irdis(cube,header,info,cube_ix):
    ''' Function for IRDIS data. Needs to:
        1. Cube the data
        2. Get all of the relevant header keywords
    ''' 
    
    filt1 = header['ESO INS1 OPTI2 NAME']
    filt2 = header['ESO INS1 FILT NAME']
    mask_name = header['ESO INS1 OPTI1 TYPE']
    if mask_name == 'MASK_SAM':
        mask_name = '7H_IRDIS'

    # IRDIS can be DPI, CI or DBI
    if filt1 == 'P0-90':
        filter_name = 'DPI_'+filt2
    elif filt1 == 'CLEAR':
        filter_name = 'CI_'+filt2
    else:
        filter_name = filt1
        
    parangs = sphere_parang(header)

    if 'ESO OBS TARG NAME' in header.keys():    
        target_name = header['ESO OBS TARG NAME']
    else:
        target_name = ''
        
    if target_name == '':
        target_name = header['OBJECT']
    if target_name == '':
        target_name = 'RA '+str(header['RA'])
    
    
    info.add_row({'filter_name':filter_name,
                  'instrument':'SPHERE',
                  'mask_name':mask_name,
                  'cube_ix':cube_ix,
                  'parallactic_angle':parangs,
                  'target_name':target_name})
    
    # If cube is only 2D, make it 3D for now
    if cube.ndim == 2:
        cube = np.reshape(cube,(1,cube.shape[0],cube.shape[1]))
    
    ## Now cube the data, extracting the two channels
    if cube.ndim == 3:
        cube = np.reshape(cube,(cube.shape[0],cube.shape[1],cube.shape[2]/2,2),order='F')
        cube = cube.transpose((3,0,1,2))
    return [cube,info]

##########
    
def preproc_ifs(cube,header,info,cube_ix):
    ''' Function for IFS data. Needs to:
        1. Cube the data
        2. Get all of the relevant header keywords
    ''' 
    
    filter_name = header['ESO INS2 OPTI2 NAME'].replace('PRI_','')
    mask_name = header['ESO INS4 OPTI14 NAME']
    if mask_name == 'ST_SAM':
        mask_name = '7H_IFS'
        
    parangs = sphere_parang(header)
    
    target_name = header['ESO OBS TARG NAME']
    if target_name == '':
        target_name = header['OBJECT']
    if target_name == '':
        target_name = 'RA '+str(header['RA'])
    
    
    info.add_row({'filter_name':filter_name,
                  'instrument':'SPHERE',
                  'mask_name':mask_name,
                  'cube_ix':cube_ix,
                  'parallactic_angle':parangs,
                  'target_name':target_name})
    
    return [cube,info]

#############

def preproc_zimpol(cube,header,info,cube_ix):
    ''' Function for ZIMPOL data. Needs to:
        1. Cube the data
        2. Get all of the relevant header keywords
    ''' 

    return [cube,info]

#############

def sphere_parang(hdr):
    """
    Reads the header and creates an array giving the paralactic angle for each frame,
    taking into account the inital derotator position.

    The columns of the output array contains:
    frame_number, frame_time, paralactic_angle
    """

    from numpy import sin, cos, arctan2, pi
    from astropy.time import Time

    r2d = 180/pi
    d2r = pi/180

    detector = hdr['HIERARCH ESO DET ID']
    if detector.strip() == 'IFS':
        offset=135.87-100.46 # from the SPHERE manual v4
    elif detector.strip() == 'IRDIS':
        #correspond to the difference between the PUPIL tracking ant the FIELD tracking for IRDIS taken here: http://wiki.oamp.fr/sphere/AstrometricCalibration (PUPOFFSET)
        offset=135.87
    else:
        offset=0
        print('WARNING: Unknown instrument in create_parang_list_sphere: '+str(detector))

    try:
        # Get the correct RA and Dec from the header
        actual_ra = hdr['HIERARCH ESO INS4 DROT2 RA']
        actual_dec = hdr['HIERARCH ESO INS4 DROT2 DEC']

        # These values were in weird units: HHMMSS.ssss
        actual_ra_hr = np.floor(actual_ra/10000.)
        actual_ra_min = np.floor(actual_ra/100. - actual_ra_hr*100.)
        actual_ra_sec = (actual_ra - actual_ra_min*100. - actual_ra_hr*10000.)

        ra_deg = (actual_ra_hr + actual_ra_min/60. + actual_ra_sec/60./60.) * 360./24.

        # the sign makes this complicated, so remove it now and add it back at the end
        sgn = np.sign(actual_dec)
        actual_dec *= sgn

        actual_dec_deg = np.floor(actual_dec/10000.)
        actual_dec_min = np.floor(actual_dec/100. - actual_dec_deg*100.)
        actual_dec_sec = (actual_dec - actual_dec_min*100. - actual_dec_deg*10000.)

        dec_deg = (actual_dec_deg + actual_dec_min/60. + actual_dec_sec/60./60.)*sgn

#        geolat_deg=float(hdr['ESO TEL GEOLAT'])
        geolat_rad=float(hdr['ESO TEL GEOLAT'])*d2r
    except:
        print('WARNING: No RA/Dec Keywords found in header')
        ra_deg=0
        dec_deg=0
#        geolat_deg=0
        geolat_rad=0

    if 'NAXIS3' in hdr:
        n_frames = hdr['NAXIS3']
    else:
        n_frames = 1

    # We want the exposure time per frame, derived from the total time from when the shutter
    # opens for the first frame until it closes at the end.
    # This is what ACC thought should be used
    # total_exptime = hdr['ESO DET SEQ1 EXPTIME']
    # This is what the SPHERE DC uses
    total_exptime = (Time(hdr['HIERARCH ESO DET FRAM UTC'])-Time(hdr['HIERARCH ESO DET SEQ UTC'])).sec
    # print total_exptime-total_exptime2
    delta_dit = total_exptime / n_frames
    dit = hdr['ESO DET SEQ1 REALDIT']

    # Set up the array to hold the parangs
    parang_array = np.zeros((n_frames))

    # Output for debugging
    hour_angles = []
    
    if ('ESO DET SEQ UTC' in hdr.keys()) and ('ESO TEL GEOLON' in hdr.keys()):
        # The SPHERE DC method
        jd_start = Time(hdr['ESO DET SEQ UTC']).jd
        lst_start = external.jd2lst(hdr['ESO TEL GEOLON'],jd_start)*3600
        # Use the old method
        lst_start = float(hdr['LST'])
    else:
        lst_start = 0.
        print('WARNING: No LST keyword found in header')


    # delta dit and dit are in seconds so we need to multiply them by this factor to add them to an LST
    time_to_lst = (24.*3600.)/(86164.1)

    if 'ESO INS4 COMB ROT' in hdr.keys() and hdr['ESO INS4 COMB ROT']=='PUPIL':

        for i in range(n_frames):

            ha_deg=((lst_start+i*delta_dit*time_to_lst + time_to_lst*dit/2.)*15./3600)-ra_deg
            hour_angles.append(ha_deg)

            # VLT TCS formula
            f1 = float(cos(geolat_rad) * sin(d2r*ha_deg))
            f2 = float(sin(geolat_rad) * cos(d2r*dec_deg) - cos(geolat_rad) * sin(d2r*dec_deg) * cos(d2r*ha_deg))
            pa = -r2d*arctan2(-f1,f2)

            pa=pa+offset

            # Also correct for the derotator issues that were fixed on 12 July 2016 (MJD = 57581)
            if hdr['MJD-OBS'] < 57581:
                alt = hdr['ESO TEL ALT']
                drot_begin = hdr['ESO INS4 DROT2 BEGIN']
                correction = np.arctan(np.tan((alt-2*drot_begin)*np.pi/180))*180/np.pi # Formula from Anne-Lise Maire
                pa += correction

            pa = ((pa + 360) % 360)
            parang_array[i] = pa

    else:
        if 'ARCFILE' in hdr.keys():
            print(hdr['ARCFILE']+' does seem to be taken in pupil tracking.')
        else:
            print('Data does not seem to be taken in pupil tracking.')

        for i in range(n_frames):
            parang_array[i] = 0

    # And a sanity check at the end
    try:
        # The parang start and parang end refer to the start and end of the sequence, not in the middle of the first and last frame.
        # So we need to correct for that
        expected_delta_parang = (hdr['HIERARCH ESO TEL PARANG END']-hdr['HIERARCH ESO TEL PARANG START']) * (n_frames-1)/n_frames
        delta_parang = (parang_array[-1]-parang_array[0]) 
        if np.abs(expected_delta_parang - delta_parang) > 1.:
            print("WARNING! Calculated parallactic angle change is >1degree more than expected!")

    except:
        pass

    return parang_array