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
from pymses.core import Filter
import numpy


class PointFunctionFilter(Filter):#{{{
	r"""
	PointFunctionFilter class

	Parameters
	----------
	mask_func : ``function``
		function evaluated to compute the data mask to apply
	source    : ``Source``
		PointDataset data source

	"""
	def __init__(self, mask_func, source):

		self.mask_func = mask_func
		Filter.__init__(self, source)

	def filtered_dset(self, dset):
		msk = self.mask_func(dset)
		return dset.filtered_by_mask(self.mask_func(dset))
#}}}

class PointIdFilter(Filter):#{{{
	r"""
	PointIdFilter class

	Parameters
	----------
	ids_to_keep : ``list`` of ``int``
		list of the particle ids to pick up
	source      : ``Source``
		PointDataset data source

	"""
	def __init__(self, ids_to_keep, source):
		self.ids_to_keep = numpy.sort(ids_to_keep)
		Filter.__init__(self, source)

	def filtered_dset(self, dset):
		ind_sort = numpy.argsort(dset["id"])
		dset = dset.reorder_points(reorder_indices=ind_sort)
		sorted_id = dset["id"]
		id_min = sorted_id[0]
		id_max = sorted_id[-1]
		mask = (self.ids_to_keep >= id_min)*(self.ids_to_keep <= id_max)
		sorted_ids_2k = self.ids_to_keep[mask]
		ind_ids = numpy.searchsorted(sorted_id, sorted_ids_2k)
		mask = (sorted_id[ind_ids] == sorted_ids_2k)
		ind_fids = ind_ids[mask]
		return dset.filtered_by_mask(ind_fids)
#}}}

class PointRandomDecimatedFilter(Filter):#{{{
	r"""
	PointRandomDecimatedFilter class
	
	Parameters
	----------
	fraction : ``float``
		fraction of the data to keep
	source   : ``Source``
		PointDataset data source

	"""
	def __init__(self, fraction, source):

		self.fraction = float(fraction)
		Filter.__init__(self, source)

	def filtered_dset(self, dset):
		mask = numpy.random.uniform(size=dset.npoints)
		mask = (mask < self.fraction)
		return dset.filtered_by_mask(mask)
#}}}


__all__ = ["PointFunctionFilter", "PointIdFilter", "PointRandomDecimatedFilter"]
