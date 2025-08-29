#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:43:41 2024

@author: roussp14
"""

import numpy as Numpy

def joinViscosity( low_v, medium_v, high_v ):
    viscosity = []
    
    for i in range(len(low_v)):
        if high_v[i] > 0:
            viscosity.append(high_v[i])
        elif medium_v[i] > 0:
            viscosity.append(medium_v[i])
        else:
            viscosity.append(low_v[i])
            
    return Numpy.array(viscosity)