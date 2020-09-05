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
from pymses.core import Filter, IsotropicExtPointDataset
from numpy import clip

class ExtendedPointFilter(Filter):
	r"""
	ExtendedParticleFilter class

	"""
	def __init__(self, source, remember_data=False, cache_dset={}):
		Filter.__init__(self, source)
		self.remember_data = remember_data
		self.cache_dset = cache_dset

	def filtered_dset(self, dset):
		r"""
		Filter a PointDataset and converts it into an IsotropicExtPointDataset with a given size for each point

		"""
		# Sizes
		s = 1./2**clip(dset["level"], 0, self.read_lmax)
		
		# Initialize the output IsotropicExtPointDataset + point size field
		pts = IsotropicExtPointDataset(dset.points, s)

		# Transfer the data arrays
		for name in dset.scalars:
			data = dset[name]
			pts.add_scalars(name, dset[name])

		for name in dset.vectors:
			data = dset[name]
			pts.add_vectors(name, dset[name])

		return pts

__all__ = ["ExtendedPointFilter"]

