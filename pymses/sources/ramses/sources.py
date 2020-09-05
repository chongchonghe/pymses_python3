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
r"""
:mod:`pymses.sources.ramses.sources` --- RAMSES data sources module
-------------------------------------------------------------------

"""
from pymses.core import Source


class RamsesGenericSource(Source):
	r"""
	RAMSES generic data source

	"""
	def __init__(self, reader_list, dom_decomp=None, cpu_list=None, ndim=None, fields_to_read=None):
		Source.__init__(self)
		self.fields_to_read = fields_to_read
		self.readers = reader_list
		ncpu = len(reader_list)
		if cpu_list == None:
			cpu_list = list(range(1,ncpu+1))
		self._data_list = cpu_list
		self.dom_decomp = dom_decomp
		self.ndim = ndim

	def get_source_type(self):
		raise NotImplementedError()

	def get_domain_dset(self, icpu, fields_to_read=None):
		r"""
		Data source reading method

		Parameters
		----------
		icpu : ``int``
			CPU file number to read
		fields_to_read : ``list`` of ``strings``
			list of AMR data fields that need to be read :
			Warning : usually this information is not needed here as it is
			already stored in self.fields_to_read (or in self.part_read_fields ):
			this option is only used internally for the remember_data cell_to_point source option
			Warning : It is the self.readers[...].ivars_descrs_by_file information that is used
			to determine which field is really read from the file system. This information is created
			when the source is created (the scripts line like source = RamsesOutput.amrsource(["rho"]))

		Returns
		-------
		dset : ``Dataset``
			the dataset containing the data from the given cpu number file

		"""
		# see RamsesOctreeReader in octree.py
		return self.readers[self._data_list.index(icpu)].read(self.read_lmax,\
							fields_to_read=fields_to_read)

class RamsesAmrSource(RamsesGenericSource):
	r"""
	RAMSES AMR data source class

	"""
	def __init__(self, reader_list, dom_decomp=None, cpu_list=None, ndim=None, fields_to_read=None):
		RamsesGenericSource.__init__(self, reader_list, dom_decomp, cpu_list, ndim, fields_to_read)

	def get_source_type(self):
		r"""
		Returns
		-------
		Source.AMR_SOURCE = 10 (compared to Source.PARTICLE_SOURCE = 11)
		
		"""
		return Source.AMR_SOURCE

class RamsesParticleSource(RamsesGenericSource):
	r"""
	RAMSES particle data source class

	"""
	def __init__(self, reader_list, dom_decomp=None, cpu_list=None, ndim=None, fields_to_read=None):
		RamsesGenericSource.__init__(self, reader_list, dom_decomp, cpu_list, ndim, fields_to_read)
	
	def get_source_type(self):
		r"""
		Returns
		-------
		Source.PARTICLE_SOURCE = 11 (compared to Source.AMR_SOURCE = 10)

		"""
		return Source.PARTICLE_SOURCE

__all__ = ["RamsesGenericSource", "RamsesAmrSource", "RamsesParticleSource"]
