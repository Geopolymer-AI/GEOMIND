#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:44:56 2024

@author: roussp14
"""

import matplotlib.pyplot as Pyplot
import numpy as Numpy
from sloot.plot import getSplineHist

def countPrecursors(precursors, precursors_names, file_name):
    precursors_count = Numpy.count_nonzero(precursors, axis = 0) / len(precursors)
    
    Pyplot.clf()
    fig, ax = Pyplot.subplots(1,1, figsize=(9,6))
    ax.grid(color='gray', linestyle='--', alpha=0.5, zorder=0)
    ax.bar( precursors_names, precursors_count* 100, zorder=3)
    Pyplot.xlabel( "Precursors classes", fontsize=10 )
    Pyplot.ylabel( "Presence in the dataset (%)", fontsize=10 )
    Pyplot.savefig( file_name, dpi = 300 )
    Pyplot.show()
    
    return fig

def propertiesHistograms(compressive_strength, viscosity, density, mixture_density, file_name, n_bins = 30):
    Pyplot.clf()
    fig, ((ax2,ax1),(ax4,ax3)) = Pyplot.subplots(2,2, figsize=(9,6), sharey='row')
    
    ax1.hist(compressive_strength, color = "tab:red", bins = n_bins, weights = Numpy.ones(len(compressive_strength)) / len(compressive_strength) * 100)
    count, bin_edges = Numpy.histogram(compressive_strength, bins = n_bins, weights = Numpy.ones(len(compressive_strength)) / len(compressive_strength) * 100)
    hist_spline, x_bins, bin_edges = getSplineHist(compressive_strength, n_bins)
    ax1.plot(x_bins, hist_spline, c = "k")
    ax1.set_xlabel( "Compressive strength (MPa)", fontsize=10 )
    ax1.set_xticks([10,30,50,70,90])
    ax1.annotate("B", xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', verticalalignment='top')
    
    log_v = Numpy.log10(viscosity)
    hist, bins = Numpy.histogram(log_v, bins = n_bins)
    ax2.hist(viscosity, color = "tab:orange", bins = 10**bins, weights = Numpy.ones(len(viscosity)) / len(viscosity) * 100 )
    hist_spline, x_bins, bin_edges = getSplineHist(log_v, n_bins)
    ax2.plot(10**x_bins, hist_spline, c = "k")
    ax2.set_xlabel( "Initial viscosity (Pa.s)", fontsize=10 )
    ax2.set_xscale('log')
    ax2.set_ylabel("Distribution (%)",fontsize=10)
    ax2.annotate("A", xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', verticalalignment='top')
    
    ax3.hist(density, color = "tab:purple", bins = n_bins, weights = Numpy.ones(len(density)) / len(density) * 100)
    hist_spline, x_bins, bin_edges = getSplineHist(density, n_bins)
    ax3.plot(x_bins, hist_spline, c = "k")
    ax3.set_xticks([1.3,1.5,1.7,1.9,2.1])
    ax3.set_xlabel( "Density (g/cm$^{3}$)", fontsize=10 )
    ax3.annotate("D", xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', verticalalignment='top')
    
    ax4.hist(mixture_density, color = "tab:brown", bins = n_bins, weights = Numpy.ones(len(mixture_density)) / len(mixture_density) * 100)
    hist_spline, x_bins, bin_edges = getSplineHist(mixture_density, n_bins)
    ax4.plot(x_bins, hist_spline, c = "k")
    ax4.set_xticks([1.4,1.6,1.8,2.0,2.2])
    ax4.set_xlabel( "Mixture density", fontsize=10 )
    ax4.set_ylabel("Distribution (%)",fontsize=10)
    ax4.annotate("C", xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', verticalalignment='top')
    
    Pyplot.tight_layout()
    Pyplot.savefig( file_name, dpi = 300 )
    Pyplot.show()
    
    return fig