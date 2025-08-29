#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:35:32 2024

@author: roussp14
"""

import matplotlib.pyplot as Pyplot
import numpy as Numpy
from pydash import times
import matplotlib.path as Mpath

#Define markers
star = Mpath.Path.unit_regular_star(6)
circle = Mpath.Path.unit_circle()
cut_star = Mpath.Path( vertices= Numpy.concatenate([circle.vertices, star.vertices[::-1, ...]]), codes= Numpy.concatenate([circle.codes, star.codes]))

def scatterExperimentalAndPredictedProperty( experimental, prediction, file_name, label = "Property", log_scale = False ):
    Pyplot.clf()
    fig, (ax1,ax2) = Pyplot.subplots(1,2, figsize=(12,4), sharey='row', gridspec_kw={'width_ratios':[1,2]})

    ax1.set_xlabel( f"Experimental {label}" )
    ax1.set_ylabel( f"Predicted {label}" )
    ax1.annotate("A", xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', verticalalignment='top')
    ax1.scatter( experimental, prediction, s = 1.5 )
    ax1.plot( [min(experimental), max(experimental)], [min(experimental), max(experimental)], color = 'red', alpha = 0.5 )
    ax1.set_box_aspect(aspect=1)
    if log_scale:
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    sorted_index = Numpy.argsort(experimental)
    experimental = experimental[sorted_index]
    prediction = prediction[sorted_index]

    ax2.set_ylabel( label )
    ax2.set_xlabel( "Number of samples" )
    ax2.annotate("B", xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', verticalalignment='top')
    ax2.scatter( times(len(experimental)), prediction, label = "Predicted", marker = star, s = 18, c = "tab:orange", edgecolors='none' )
    ax2.scatter( times(len(experimental)), experimental, label = "Experimental", marker = cut_star, s = 18, c = "tab:blue", edgecolors='none' )
    ax2.legend( loc="upper center" )

    Pyplot.tight_layout()
    Pyplot.savefig( file_name, dpi = 300 )
    Pyplot.show()
    
    return fig