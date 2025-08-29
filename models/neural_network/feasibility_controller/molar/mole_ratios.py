#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:39:47 2024

@author: roussp14
"""

import tensorflow as Tensorflow

SIO2_S1 = 14.32
SIO2_S3 = 18.68
SIO2_SNA = 27.50
SIO2_S3p = 23.4
SIO2_M1 = 55
SIO2_M2 = 55
SIO2_M3 = 54
SIO2_M4 = 52.4
SIO2_M5 = 59.9
K2O_S1 = 6.41
K2O_S3 = 21.92
K2O_SNA = 8.30
K2O_S3p = 21.7
K2O_KOH = 0.857
K2O_NAOH = 0.97
ALO_M1 = 40
ALO_M2 = 39
ALO_M3 = 46
ALO_M4 = 45.3
ALO_M5 = 35.3

"""
getMoleRatios

This function is used in order to calculate molar ratios between silica, alumina and water from precursors.

Arguments:
    precursors_tensor     2D-Tensor from Tensorflow or Keras
Returns:
    (siM_sol, siAl, solidLiquid)    Molar ratios: Silica of solutions / Potassium, Silica / Alumina, Solid / Liquid 
"""
def getMoleRatios( precursors_tensor ):
    s1, s3, sna, s3p, koh, naoh, m1, m2, m3, m4, m5 =  Tensorflow.split( precursors_tensor, 11, -1 )
    
    m_sol = ( s1 * K2O_S1 + s3 * K2O_S3 + s3p * K2O_S3p ) / 4710 + ( sna * K2O_SNA ) / 3098.5
    m_moh = ( koh * K2O_KOH ) / 56.1 + ( naoh * K2O_NAOH ) / 39.99
    
    si_sol = ( s1 * SIO2_S1 + s3 * SIO2_S3 + sna * SIO2_SNA + s3p * SIO2_S3p ) / 6008 
    m_tot = m_sol + m_moh
    
    solid = m1 + m2 + m3 + m4 + m5
    liquid = s1 + s3 + sna + s3p + koh + naoh
    
    al_tot = ( m1 * ALO_M1 + m2 * ALO_M2 + m3 * ALO_M3 + m4 * ALO_M4 + m5 * ALO_M5 ) / 5100
    si_met = ( m1 * SIO2_M1 + m2 * SIO2_M2 + m3 * SIO2_M3 + m4 * SIO2_M4 + m5 * SIO2_M5 ) / 6008
    si_tot = si_met + si_sol
    
    tot = ( si_tot + al_tot + m_tot )
    si = si_tot / tot
    al = al_tot / tot
    
    siM_sol =  Tensorflow.where( m_tot > 0, si_sol / m_tot, 0. )
    siAl = Tensorflow.where( al > 0, si / al, 0. )
    solidLiquid = Tensorflow.where( liquid > 0, solid / liquid, 0. )
    
    return siM_sol, siAl, solidLiquid