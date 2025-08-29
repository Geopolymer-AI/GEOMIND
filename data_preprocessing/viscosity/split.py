#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:43:08 2024

@author: roussp14
"""

import numpy as Numpy

def splitViscosity( viscosity, limit_low, limit_high, mean, v_min = None, v_max = None ):
    v = -Numpy.ones( (len(viscosity), 3) )
    
    for i in range(len(viscosity)):
        value = viscosity[i]
        
        if value == 0:
            value = mean
            
        if value > limit_high:
            v[i,2] = value
        elif value < limit_low:
            v[i,0] = value
        else:
            v[i,1] = value
    
    if v_min is None:
        v_min = Numpy.min(Numpy.where(v < 0, 1e10, v), axis = 0)
    if v_max is None:
        v_max = Numpy.max(v, axis = 0)
    
    for i in range(len(viscosity)):
        v[i] = (v[i] - v_min) / (v_max - v_min)

        v = Numpy.where( v < 0, -1, v )
            
    return v, v_min, v_max