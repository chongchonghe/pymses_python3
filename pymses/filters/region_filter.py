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
from pymses.core import Source, SubsetFilter


class RegionFilter(SubsetFilter):#{{{
	r"""
	Region Filter class. Filters the data contained in a given region 
	of interest.

	Parameters
	----------
	region : :class:`~pymses.utils.regions.Region`
		region of interest 
	source : :class:`~pymses.core.sources.Source`
		data source

	"""
	def __init__(self, region, source):
		self.region = region
		if source.dom_decomp is not None:
			map_list = source.dom_decomp.map_region(self.region)
		else:
			map_list = None
		SubsetFilter.__init__(self, map_list, source)
	
	def filtered_dset(self, dset):
		if (self.source.get_source_type() == Source.AMR_SOURCE):
			# update the active mask
			grid_mask = dset.get_active_mask()
			grid_center = dset.amr_struct["grid_centers"][grid_mask, :]
			filter_mask = self.region.contains(grid_center)
			dset.active_mask[grid_mask] = grid_mask[grid_mask]*filter_mask
			return dset
		else:
			return dset.filtered_by_mask(
				self.region.contains(dset.points))
			
	
#}}}

__all__ = ["RegionFilter"]
