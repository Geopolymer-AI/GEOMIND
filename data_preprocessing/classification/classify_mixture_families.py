#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:39:57 2024

@author: roussp14
"""

import numpy as Numpy

npand = Numpy.logical_and
npor = Numpy.logical_or

def splitMixtures(mixtures, cond):
  return [mixtures[cond], mixtures[~cond]]

def isS1(precursors):
    return precursors[:,0] > 0

def isS3(precursors):
    return precursors[:,1] > 0

def isSna(precursors):
    return precursors[:,2] > 0

def isS3p(precursors):
    return precursors[:,3] > 0

def isKoh(precursors):
    return precursors[:,4] > 0

def isNaoh(precursors):
    return precursors[:,5] > 0

def isM1(precursors):
    return precursors[:,6] > 0

def isM2(precursors):
    return precursors[:,7] > 0

def isM3(precursors):
    return precursors[:,8] > 0

def isM4(precursors):
    return precursors[:,9] > 0

def isM5(precursors):
    return precursors[:,10] > 0

def isM1Major(precursors):
    return npand(isM1(precursors), npand(precursors[:,6] > precursors[:,7], npand(precursors[:,6] > precursors[:,8], npand(precursors[:,6] > precursors[:,9], precursors[:,6] > precursors[:,10]))))

def isM2Major(precursors):
    return npand(isM2(precursors), npand(precursors[:,7] > precursors[:,6], npand(precursors[:,7] > precursors[:,8], npand(precursors[:,7] > precursors[:,9], precursors[:,7] > precursors[:,10]))))

def isM3Major(precursors):
    return npand(isM3(precursors), npand(precursors[:,8] > precursors[:,7], npand(precursors[:,8] > precursors[:,6], npand(precursors[:,8] > precursors[:,9], precursors[:,8] > precursors[:,10]))))

def isM4Major(precursors):
    return npand(isM4(precursors), npand(precursors[:,9] > precursors[:,7], npand(precursors[:,9] > precursors[:,8], npand(precursors[:,9] > precursors[:,6], precursors[:,9] > precursors[:,10]))))

def isM5Major(precursors):
    return npand(isM5(precursors), npand(precursors[:,10] > precursors[:,7], npand(precursors[:,10] > precursors[:,8], npand(precursors[:,10] > precursors[:,9], precursors[:,10] >= precursors[:,6]))))

def isS1Major(precursors):
    return npand(isS1(precursors), npand(precursors[:,0] > precursors[:,1], npand(precursors[:,0] > precursors[:,2], precursors[:,0] > precursors[:,3])))

def isS3Major(precursors):
    return npand(isS3(precursors), npand(precursors[:,1] > precursors[:,0], npand(precursors[:,1] > precursors[:,2], precursors[:,1] > precursors[:,3])))

def isSnaMajor(precursors):
    return npand(isSna(precursors), npand(precursors[:,2] > precursors[:,1], npand(precursors[:,2] > precursors[:,0], precursors[:,2] > precursors[:,3])))

def isS3pMajor(precursors):
    return npand(isS3p(precursors), npand(precursors[:,3] > precursors[:,1], npand(precursors[:,3] > precursors[:,2], precursors[:,3] > precursors[:,0])))

def classifyDataByMixtureFamilies( precursors, properties ):
    nrs_nrm, others1 = splitMixtures(precursors, npand(isS1Major(precursors), npor(isM1Major(precursors), isM5Major(precursors))))
    properties_nrs_nrm, properties_others1 = splitMixtures(properties, npand(isS1Major(precursors), npor(isM1Major(precursors), isM5Major(precursors))))
    nrs_rm, others2 = splitMixtures(others1, isS1Major(others1))
    properties_nrs_rm, properties_others2 = splitMixtures(properties_others1, isS1Major(others1))
    rs_nrm1, others3 = splitMixtures(others2, isM1Major(others2))
    properties_rs_nrm1, properties_others3 = splitMixtures(properties_others2, isM1Major(others2))
    rs_nrm5, rs_rm = splitMixtures(others3, isM5Major(others3))
    properties_rs_nrm5, properties_rs_rm = splitMixtures(properties_others3, isM5Major(others3))
    
    return [{"nrs_nrm": nrs_nrm, "nrs_rm": nrs_rm, "rs_nrm1": rs_nrm1, "rs_nrm5": rs_nrm5, "rs_rm": rs_rm}, {"nrs_nrm": properties_nrs_nrm, "nrs_rm": properties_nrs_rm, "rs_nrm1": properties_rs_nrm1, "rs_nrm5": properties_rs_nrm5, "rs_rm": properties_rs_rm}]
