#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:36:10 2024

@author: roussp14
"""

from sloot.plot import hist2D, heatmap
from data_preprocessing import classifyDataByMixtureFamilies
import numpy as Numpy

def propertiesHistograms2D(compressive_strength, viscosity, density, mixture_density, folder, n_bins = 30):
    specific_compressive_strength = compressive_strength / density
    
    hist2D(viscosity,compressive_strength, "Initial viscosity (Pa.s)", "Compressive strength (MPa)", f"{folder}/compression_viscosite.png", x_color = "tab:orange", y_color = "tab:red", nxbins = n_bins, nybins = n_bins, nbins = 500, x_scale = "log")
    
    hist2D(density,compressive_strength, "Density (g/cm$^{3}$)", "Compressive strength (MPa)", f"{folder}/compression_masse_volumique.png", x_color = "tab:purple", y_color = "tab:red", nxbins = n_bins, nybins = n_bins, nbins = 500)
    hist2D(mixture_density,compressive_strength, "Mixture density", "Compressive strength (MPa)", f"{folder}/compression_densite.png", x_color = "tab:brown", y_color = "tab:red", nxbins = n_bins, nybins = n_bins, nbins = 500)
    
    hist2D(viscosity, density, "Initial viscosity (Pa.s)", "Density (g/cm$^{3}$)", f"{folder}/masse_volumique_viscosite.png", x_color = "tab:orange", y_color = "tab:purple", nxbins = n_bins, nybins = n_bins, nbins = 500, x_scale = "log")
    hist2D(viscosity, mixture_density, "Initial viscosity (Pa.s)", "Mixture density", f"{folder}/densite_viscosite.png", x_color = "tab:orange", y_color = "tab:brown", nxbins = n_bins, nybins = n_bins, nbins = 500, x_scale = "log")
    
    hist2D(density, mixture_density, "Density (g/cm$^{3}$)", "Mixture density", f"{folder}/densite_masse_volumique.png", x_color = "tab:purple", y_color = "tab:brown", nxbins = n_bins, nybins = n_bins, nbins = 500)
    
    hist2D(specific_compressive_strength,compressive_strength, "Specific compr. strength (MPa.cm$^{3}$/g)", "Compressive strength (MPa)", f"{folder}/compression_specific_compression.png", x_color = "tab:red", y_color = "tab:red", nxbins = n_bins, nybins = n_bins, nbins = 500)
    
    hist2D(viscosity, specific_compressive_strength, "Initial viscosity (Pa.s)", "Specific compr. strength (MPa.cm$^{3}$/g)", f"{folder}/specific_compression_viscosite.png", x_color = "tab:orange", y_color = "tab:red", nxbins = n_bins, nybins = n_bins, nbins = 500, x_scale = "log")
    hist2D(density, specific_compressive_strength, "Density (g/cm$^{3}$)", "Specific compr. strength (MPa.cm$^{3}$/g)", f"{folder}/specific_compression_masse_volumique.png", x_color = "tab:purple", y_color = "tab:red", nxbins = n_bins, nybins = n_bins, nbins = 500)
    hist2D(specific_compressive_strength, mixture_density, "Specific compr. strength (MPa.cm$^{3}$/g)", "Mixture density", f"{folder}/densite_specific_compression.png", x_color = "tab:red", y_color = "tab:brown", nxbins = n_bins, nybins = n_bins, nbins = 500)

def propertiesHeatmap(compressive_strength, viscosity, density, file_name, should_scatter = True):
    return heatmap(density, viscosity, compressive_strength, 'Density (g/cm$^{3}$)', "Initial viscosity (Pa.s)", 'Compressive strength (MPa)', file_name, y_scale = 'log', should_scatter = should_scatter )
    
def propertiesHeatmapsWithMixtureFamilies(properties, precursors, folder, properties_indexes = [1,0,2], labels = ['Density (g/cm$^{3}$)', "Initial viscosity (Pa.s)", 'Compressive strength (MPa)']):
    if not (isinstance(properties_indexes, list) and len(properties_indexes) == 3):
        raise TypeError( "properties_indexes must be a list length of 3" )
    
    if not (isinstance(labels, list) and len(labels) == 3):
        raise TypeError( "labels must be a list length of 3" )

    i, j, k = properties_indexes
    x, y, z = [properties[:,i], properties[:,j], properties[:,k]]
    precursors, properties = classifyDataByMixtureFamilies(precursors, properties)

    areas = [
        [properties["nrs_nrm"][:,i], properties["nrs_nrm"][:,j], properties["nrs_nrm"][:,k]],
        [properties["nrs_rm"][:,i], properties["nrs_rm"][:,j], properties["nrs_rm"][:,k]],
        [properties["rs_nrm1"][:,i], properties["rs_nrm1"][:,j], properties["rs_nrm1"][:,k]],
        [properties["rs_nrm5"][:,i], properties["rs_nrm5"][:,j], properties["rs_nrm5"][:,k]],
        [properties["rs_rm"][:,i], properties["rs_rm"][:,j], properties["rs_rm"][:,k]]]

    areas_labels = ["NRS-NRM", "NRS-RM", "RS-NRM1", "RS-NRM5", "RS-RM"]

    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    z_min = z.min()
    z_max = z.max()
    
    heatmap(areas[0][0], areas[0][1], areas[0][2], labels[0], labels[1], labels[2], f"{folder}/heatmap_NRS_NRM.png", label = "Non reactive solution (S1)\n+ metakaolin (M1 and/or M5)", xmin = x_min, xmax = x_max, ymin = Numpy.log10(y_min), ymax = Numpy.log10(y_max), zmin = z_min, zmax = z_max, y_scale = 'log' )
    heatmap(areas[1][0], areas[1][1], areas[1][2], labels[0], labels[1], labels[2], f"{folder}/heatmap_NRS_RM.png", label = "Non reactive solution (S1)\n+ reactive metakaolin (M2 and/or M3 and/or M4)", xmin = x_min, xmax = x_max, ymin = Numpy.log10(y_min), ymax = Numpy.log10(y_max), zmin = z_min, zmax = z_max, y_scale = 'log' )
    heatmap(areas[2][0], areas[2][1], areas[2][2], labels[0], labels[1], labels[2], f"{folder}/heatmap_RS_NRM1.png", label = "Reactive solution (S3 and/or S3' and/or SNa)\n+ non reactive metakaolin (M1)", xmin = x_min, xmax = x_max, ymin = Numpy.log10(y_min), ymax = Numpy.log10(y_max), zmin = z_min, zmax = z_max, y_scale = 'log' )
    heatmap(areas[3][0], areas[3][1], areas[3][2], labels[0], labels[1], labels[2], f"{folder}/heatmap_RS_NRM5.png", label = "Reactive solution (S3 and/or S3' and/or SNa)\n+ non reactive metakaolin (M5)", xmin = x_min, xmax = x_max, ymin = Numpy.log10(y_min), ymax = Numpy.log10(y_max), zmin = z_min, zmax = z_max, y_scale = 'log' )
    heatmap(areas[4][0], areas[4][1], areas[4][2], labels[0], labels[1], labels[2], f"{folder}/heatmap_RS_RM.png", label = "Reactive solution (S3 and/or S3' and/or SNa)\n+ reactive metakaolin (M2 and/or M3 and/or M4)", xmin = x_min, xmax = x_max, ymin = Numpy.log10(y_min), ymax = Numpy.log10(y_max), zmin = z_min, zmax = z_max, y_scale = 'log' )
    
    heatmap(x, y, z, labels[0], labels[1], labels[2], f"{folder}/heatmap.png", y_scale = 'log')
    return heatmap(x, y, z, labels[0], labels[1], labels[2], f"{folder}/heatmap_with_families.png", y_scale = 'log', areas = areas, areas_labels = areas_labels)