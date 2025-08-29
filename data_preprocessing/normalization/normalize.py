#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:41:25 2024

@author: roussp14
"""

import numpy as Numpy

def getFloats( data ):
    floats = []
    for value in data:
        if value == '':
            continue
        floats.append(value.replace(',', '.'))
        
    return Numpy.array(floats).astype(Numpy.double)


def formatFloats( data, mean, std, should_standardize = True, should_logarithm = False ):
    formatted = []
    for value in data:
        if value == '':
            value = 0
        else:   
            value = float(value.replace(',', '.'))
            if should_logarithm:
                value = value if value > 0 else 1e-6
                value = Numpy.log(value)
            if should_standardize:
                value = ( value - mean ) / std
            
        formatted.append(value)
        
    return Numpy.array(formatted).astype(Numpy.double)

def normalizeData( data, mean = None, std = None, should_standardize = True, should_logarithm = False ):
    floats = getFloats(data)
    
    if should_logarithm:
        floats = Numpy.where( floats > 0, floats, 1e-5 )
        floats = Numpy.log(floats)
    if mean is None:
        mean = floats.mean()
    if std is None:
        std = floats.std()
        
    data = formatFloats(data, mean, std, should_standardize, should_logarithm)

    return data, mean, std