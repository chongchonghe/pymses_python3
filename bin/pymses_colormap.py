#!/usr/bin/env python
# -*- coding: utf-8 -*-
# colormap map testing script
from pymses.analysis.visualization.image_plot_utils import *
from optparse import OptionParser
parser = OptionParser()
parser.usage = "%prog pymses_colormap.py HDF5fileName"
(opts, args) = parser.parse_args()
HDF5fileName= args[0]	
import pylab as m
# User defined colormap :
cdict = { # (0 -> black), (0.2 -> blue), (0.5 -> yellow), (1 -> red) 
'red'  :  ((0., 0., 0.), (0.2, 0., 0.), (0.5, 0.9, .9), (1., 1., 1.)),
'green':  ((0., 0., 0.), (0.2, 0.0, 0.0), (0.5, .9, .9), (1., 0., 0.)),
'blue' :  ((0., 0., 0.), (0.2, .8, .8), (0.5, 0.4, 0.4), (1., 0., 0.))
}
#generate the colormap with 1e5 interpolated values
my_cmap = m.matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1e5)
# my_cmap = "jet" # matplotlib already defined colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
save_HDF5_to_img(HDF5fileName, cmap=my_cmap, img_path="./")
