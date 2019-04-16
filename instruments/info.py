#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:24:14 2019

@author: cheetham
"""

from astropy.table import Table
import numpy as np

def new_info():
    ''' Common definition of the info structure so that we know what it contains
    regardless of instrument
    '''
#    info = {'instrument':n_cubes*[''],
#            'mask_name':n_cubes*[''],
#            'parallactic_angle':n_cubes*[[]],
#            'filter_name':n_cubes*['']}
    
    info = Table(names=['cube_ix','target_name','instrument','mask_name',
                        'filter_name','parallactic_angle'],
                 dtype=[int,'S40','S40','S40',
                        'S40',np.ndarray])
    
    return info

