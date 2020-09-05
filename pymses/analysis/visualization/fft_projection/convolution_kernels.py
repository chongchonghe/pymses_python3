# License:
#   Copyright (C) 2011 Thomas GUILLET, Damien CHAPON, Marc LABADENS. All Rights Reserved.
#
#   This file is part of PyMSES.
#
#   PyMSES is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   PyMSES is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with PyMSES.  If not, see <http://www.gnu.org/licenses/>.
import numpy as N

class ConvolKernel(object):#{{{
	"""Convolution kernel class
	"""

	def __init__(self, ker_func, size_func=None, max_size=None):
		"""Convolution kernel builder

		ker_func : convolution kernel function => 2D function lambda x, y, size: f(x, y, size)
		size     : kernel size factor. The size of the convolution kernel is set to 'size' x the local leaf-cell size.
		max_size : maximum size of the convolution kernel.
		"""
		# Kernel 2D function
		self.ker_func = ker_func

		# Size function
		if size_func is None:
			self.size_func = lambda dset: dset.get_sizes()
		else:
			self.size_func = size_func

		# Max. size of the convolution kernel
		self.max_size = max_size
		
		# Allow to change the size factor without having to create a new kernel object
		self.FFTkernelSizeFactor = 1

	def get_size(self, dset):
		"""
		"""
		return self.FFTkernelSizeFactor * self.size_func(dset)
		
	def get_max_size(self):
		return self.FFTkernelSizeFactor * self.max_size

	def get_convolved_axes(self):
		raise NotImplementedError

	def convol_fft(self, map_dict, cam_dict):
		"""FFT convolution method designed to convolute a dict. of maps into
		a single map

		map_dict : map dict. where the dict. keys are the size of the convolution kernel.
		cam_dict : ExtendedCamera dict. corrsponding to the different maps of the map dict.
		"""
		kernel_sizes = N.sort(list(map_dict.keys()))[::-1]
		print("Processing FFT-convolution (%i different kernel sizes)"%kernel_sizes.size)
		map = None
		for kernel_size in kernel_sizes:
			# Extended camera
			c = cam_dict[kernel_size]

			# Map coordinates edges
			xedges, yedges = c.get_pixels_coordinates_edges()
			xc = (xedges[1:] + xedges[:-1])/2.
			yc = (yedges[1:] + yedges[:-1])/2.

			# Region of interest limits
			imin, imax, jmin, jmax = c.get_window_mask()

			level = int(N.log2(1.0/kernel_size))
			print(" -> level = %i"%level)
			kernel = self.ker_func(xc, yc, kernel_size)
			if N.max(kernel) != 0.0:
				kernel = N.fft.fftshift(kernel / N.sum(kernel))
				kernel = N.fft.fftn(kernel)
				conv = kernel * N.fft.fftn(map_dict[kernel_size])
				del kernel
				dat = N.fft.ifftn(conv)
				del conv
				dat = N.real(dat)
				if map is None:
					map = dat[imin:imax,jmin:jmax]
				else:
					map = map + dat[imin:imax, jmin:jmax]
				del dat
			else:
				print("WARNING : this kernel is too small to be taken into account on this map : is a point size filter properly set?")
			
		return map
#}}}

class GaussSplatterKernel(ConvolKernel):#{{{
	"""2D Gaussian splatter convolution kernel
	"""
	def __init__(self, size_func=None, max_size=None):
		"""2D Gaussian splatter convolution kernel builder
		"""
		f = lambda x, y, size: 1./(2.*N.pi*size**2)*N.outer(N.exp(-x**2/(0.5*size**2)),N.exp(-y**2/(0.5*size**2)))
		ConvolKernel.__init__(self, f, size_func, max_size)
	
	def get_convolved_axes(self):
		return [0, 1]
#}}}

class Cos2SplatterKernel(ConvolKernel):#{{{
	"""2D Squared cosine splatter convolution kernel
	"""
	def __init__(self, size_func=None, max_size=None):
		"""2D Squared cosine splatter convolution kernel builder
		"""
		def f(x, y, size):
			ker = N.outer(N.cos(N.pi * x / (2.*size))**2,N.cos(N.pi * y / (2.*size))**2)
			ker[(N.abs(x)>size),:]=0.0
			ker[:,(N.abs(y)>size)]=0.0
			return ker
		ConvolKernel.__init__(self, f, size_func, max_size)
	
	def get_convolved_axes(self):
		return [0, 1]
#}}}

class PyramidSplatterKernel(ConvolKernel):#{{{
	"""2D pyramidal splatter convolution kernel
	"""
	def __init__(self, size_func=None, max_size=None):
		"""2D pyramidal splatter convolution kernel builder
		"""
		def f(x, y, size):
			ker = N.outer(N.abs(1.0 - x / size), N.abs(1.0 - y / size))
			ker[(N.abs(x)>size),:]=0.0
			ker[:,(N.abs(y)>size)]=0.0
			return ker
		ConvolKernel.__init__(self, f, size_func, max_size)
	
	def get_convolved_axes(self):
		return [0, 1]
#}}}

class Gauss1DSplatterKernel(ConvolKernel):#{{{
	"""2D Gaussian splatter convolution kernel
	"""
	def __init__(self, axis, size_func=None, max_size=None):
		"""2D Gaussian splatter convolution kernel builder
		"""
		assert axis in [0, 1], "axis param must be in [0, 1]."
		if axis == 0:
			def f(x, y, size):
				yvect = N.zeros_like(y)
				yvect[(N.abs(y)==N.min(N.abs(y)))] = 1.0
				ker = N.outer(N.exp(-x**2/(0.5*size**2)),yvect)
				return ker
		else:
			def f(x,y,size):
				xvect = N.zeros_like(x)
				xvect[(N.abs(x)==N.min(N.abs(x)))] = 1.0
				ker = N.outer(xvect,N.exp(-y**2/(0.5*size**2)))
				return ker
		ConvolKernel.__init__(self, f, size_func, max_size)
	
	def get_convolved_axes(self):
		return [axis]
#}}}

__all__ = ["ConvolKernel",
		   "GaussSplatterKernel",
		   "Gauss1DSplatterKernel",
		   "PyramidSplatterKernel",
		   "Cos2SplatterKernel"]
