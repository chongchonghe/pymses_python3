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
"""
:mod:`pymses.core.datasets` --- PyMSES generic dataset module
=============================================================

"""
import numpy
try:
	import tables
except:
	print("WARNING : Can't import tables module...")
from pymses.utils.misc import concatenate_reorder
from .sources import Source


class Dataset(Source):#{{{
	r"""
	Base class for all dataset objects

	"""

	def __init__(self):
		self._all_data = {}
		self.vectors = []
		self.scalars = []
		Source.__init__(self)
		self._data_list = [0]

	@property
	def fields(self):
		r"""
		Dictionary of the fields in the dataset
		
		"""
		return self._all_data

	def __getitem__(self, item):
		"Quick access to fields"
		return self._all_data[item]
	
	def add_scalars(self, name, data):
		r"""
		Scalar field addition method

		Parameters
		----------
		name : ``string``
			human-readable name of the scalar field to add
		data : ``array``
			raw data array of the new scalar field

		"""
		assert name not in self._all_data, \
				"cannot have duplicate data name in dataset"

		self._all_data[name] = data
		self.scalars.append(name)


	def add_vectors(self, name, data):
		r"""
		Vector field addition method

		Parameters
		----------
		name : ``string``
			human-readable name of the vector field to add
		data : ``array``
			raw data array of the new vector field

		"""
		assert name not in self._all_data, \
				"cannot have duplicate data name in dataset"

		self._all_data[name] = data
		self.vectors.append(name)

	def get_domain_dset(self, idomain, fields_to_read=None):
		return self

	def iter_dsets(self):
		r"""
		Returns an iterator over itself

		"""
		return iter([self])
	
	def write_hdf5(self, h5file, where="/", close_at_end=False):
		r"""

		"""
		if isinstance(h5file, tables.file.File):
			h5fobj = h5file
			close_at_end = False
		else:
			h5fobj = tables.openFile(h5file, "w")
			close_at_end = True

		def process(kind, kind_vars):
			group = h5fobj.createGroup(where, kind)

			for name in kind_vars:
				data = self[name]
				h5fobj.createArray(group, name, data)

		# Write the data
		process("vectors", self.vectors)
		process("scalars", self.scalars)

		# Close if needed
		if close_at_end:
			h5fobj.close()

	@classmethod
	def from_hdf5(cls, h5file, where="/", close_at_end=False):
		r"""

		"""
		if isinstance(h5file, tables.file.File):
			h5fobj = h5file
		else:
			h5fobj = tables.openFile(h5file, "r")

		# PointDataset initialisation
		dset = cls()

		_read_fields_from_HDF5(dset, h5fobj, where)

		# Close HDF5 file if needed
		if close_at_end:
			h5fobj.close()

		return dset


#}}}

def _read_fields_from_HDF5(dset, h5file_obj, where):#{{{
	for kind, add_func in [("scalars", dset.add_scalars),("vectors", dset.add_vectors)]:
		group = h5file_obj.getNode(where, kind)
		for arr in h5file_obj.listNodes(group):
			data = arr.read()
			name = arr.name
			add_func(name, data)
#}}}

class PointDataset(Dataset):#{{{
	r"""
	Point-based dataset base class

	"""
	def __init__(self, points):
		Dataset.__init__(self)
		self.points = numpy.asarray(points)
		self.npoints = self.points.shape[0]
		Dataset.__init__(self)
		self.random_shift = False

	def get_source_type(self):
		return Source.PARTICLE_SOURCE

	def transform(self, xform):
		r"""
		Transform the dataset according to the given `xform` :class:`Transformation<pymses.core.transformations.Transformation>`

		Parameters
		----------
		xform : :class:`Transformation<pymses.core.transformations.Transformation>`
		"""
		# Do nothing to the scalars
		pass

		# Transform the vectors at the points self.points
		for vname in self.vectors:
			self._all_data[vname] = xform.transform_vectors(self._all_data[vname], self.points)

		# Transform the points
		self.points = xform.transform_points(self.points)


	@classmethod
	def concatenate(cls, dsets, reorder_indices=None):
		r"""
		Datasets concatenation class method. Return a new dataset

		Parameters
		----------
		dsets : ``list`` of ``PointDataset``
			list of all datasets to concatenate
		reorder_indices : ``array`` of ``int`` (default to None)
			particles reordering indices

		Returns
		-------
		dset : the new created concatenated ``PointDataset``

		"""
		# Prefilter the dsets list to remove any None
		dsets = [ds for ds in dsets if ds is not None]

		if dsets == []:
			return None

		# Get the concatenated points and build the new dataset
		dset_points = [ds.points for ds in dsets]
		new_points = concatenate_reorder(dset_points, reorder_indices, axis=0)
		new_dset = cls(new_points)

		dset0 = dsets[0]
		data_names = list(dset0.fields.keys())

		def process(kind_vars, add_func):
			for name in kind_vars:

				# Get the same data across all datasets into a list
				data_list = [ ds[name] for ds in dsets ]

				# Concatenate, possibly with reordering
				new_data = concatenate_reorder(data_list, reorder_indices, axis=0)

				# Add to dataset with the proper kind
				add_func(name, new_data)

		# Process the data
		process(dset0.vectors, new_dset.add_vectors)
		process(dset0.scalars, new_dset.add_scalars)

		return new_dset
	
	def add_random_shift(self):
		r"""
		Add a random shift to point positions in order to avoid grid alignment
		effect on processed images. The field "size" (from CellsToPoints Filter
		and IsotropicExtPointDataset) is needed to know the shift amplitude. This
		method is processed only once, and turn the random_shift attribute to True.
		"""
		if not self.random_shift:
			s = numpy.zeros((self.npoints,3))
			s[:,0] = self.fields["size"]
			s[:,1] = self.fields["size"]
			s[:,2] = self.fields["size"]
			self.points += (numpy.random.rand(self.npoints,3)-.5) * s
			self.random_shift = True

	def copy(self):
		c = self.__class__(self.points.copy())
		for key in self.scalars:
			c.add_scalars(key, self[key].copy())
		for key in self.vectors:
			c.add_vectors(key, self[key].copy())
		return c

	def reorder_points(self, reorder_indices):
		r"""
		Datasets reorder method. Return a new dataset

		Parameters
		----------
		reorder_indices : ``array`` of ``int``
			points order indices

		Returns
		-------
		dset : the new created reordered ``PointDataset``

		"""
		pts = self.points.copy()
		new_dset = self.__class__(pts[reorder_indices])
		
		def process(kind_vars, add_func):
			for name in kind_vars:
				add_func(name, self[name][reorder_indices])

		# Process the data
		process(self.vectors, new_dset.add_vectors)
		process(self.scalars, new_dset.add_scalars)

		return new_dset

	def filtered_by_mask(self, mask_array):
		r"""
		Datasets filter method. Return a new dataset

		Parameters
		----------
		mask_array : ``numpy.array`` of ``numpy.bool``
			filter mask

		Returns
		-------
		dset : the new created filtered ``PointDataset``

		"""
		if self.npoints == 0:
			# Don't attempt to filter an empty data set
			return self

		new_dset = self.__class__(self.points[mask_array])


		def process(kind_vars, add_func):
			for name in kind_vars:
				data = self[name]
				add_func(name, data[mask_array])

		# Reconstruct data fields with filtered data
		process(self.scalars, new_dset.add_scalars)
		process(self.vectors, new_dset.add_vectors)

		return new_dset


	def average_point(self, weight_func=None):
		w = None
		if weight_func is not None:
			w = weight_func(self)

		try:
			p, sow = numpy.average(self.points, axis=0, weights=w, returned=True)
		except:
			p = numpy.zeros(self.points.shape[1])
			sow = 0.

		return (p, sow)


	def write_hdf5(self, h5file, where="/"):
		r"""

		"""
		if isinstance(h5file, tables.file.File):
			h5fobj = h5file
			close_at_end = False
		else:
			h5fobj = tables.openFile(h5file, "w")
			close_at_end = True

		# Write the points
		h5fobj.createArray(where, "points", self.points)
		
		Dataset.write_hdf5(self, h5fobj, where)

		# Close if needed
		if close_at_end:
			h5fobj.close()

	@classmethod
	def from_hdf5(cls, h5file, where="/"):
		r"""

		"""
		if isinstance(h5file, tables.file.File):
			h5fobj = h5file
			close_at_end = False
		else:
			h5fobj = tables.openFile(h5file, "r")
			close_at_end = True

		# Read the points
		p = h5fobj.getNode(where, "points").read()
		
		# PointDataset initialisation
		pdset = cls(p)

		_read_fields_from_HDF5(pdset, h5fobj, where)

		# Close HDF5 file if needed
		if close_at_end:
			h5fobj.close()

		return pdset
#}}}

class IsotropicExtPointDataset(PointDataset):#{{{
	r"""
	Extended point dataset class

	"""
	def __init__(self, points, sizes=None):
		PointDataset.__init__(self, points)
		if sizes is not None:
			self.add_scalars("size", sizes)

	def get_sizes(self):
		r"""
		Returns
		-------
		sizes : ``array``
			point sizes array

		"""
		return self._all_data["size"]
#}}}

__all__ = ["Dataset", "PointDataset", "IsotropicExtPointDataset"]
