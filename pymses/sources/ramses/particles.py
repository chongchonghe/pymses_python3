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
import os
import numpy
from . import _read_ramses
from . import filename_utils
from pymses.core.datasets import PointDataset


def read_ramses_particle_file(part_fname, lbox):
	"""
	Reads a RAMSES .part file into memory

	Parameters
	----------

	part_fname : ``string``
		filename of the .part file
	lbox : ``float``
		simulation box width
	Returns
	-------
	header_dict : ``dict``
		particle file header dictionary
	part_dict : ``dict``
		particle data dictionary

	"""

	header, part = _read_ramses.read_parts(part_fname, lbox)

	# Extract icpu from filename
	try:
		icpu = int(os.path.basename(part_fname).split("out")[1])
	except ValueError:
		raise RuntimeError("cannot parse icpu from file name")
	header["icpu"] = icpu

	return header, part



class RamsesParticleReader():
	""" RAMSES particle file reader
	"""
	def __init__(self, output_repos, iout, icpu, part_read_fields, lbox,
			fortran_file_kwargs={}, select_stars=True,
			select_dark_matter=True, verbose=True):
		"""
		Parameters
		----------
		output_repos              : :  ``string``
			path to output repository
		iout          : : ``int`` 
			output number
		part_read_fields              : :  ``list of string``
			fields to read like ["mass", "level", "epoch"]
		lbox          : : ``float``
			box length (particules positions are divided by that value)
		fortran_file_kwargs :  ``dictionary`` (default {})
			useless here
		select_stars :  ``boolean`` (default True)
			if True :  select and read STARS particules
				(with "epoch" field != 0)
		select_dark_matter :  ``boolean`` (default True)
			if True : select and read only DARK MATTER particules
				(with "epoch" field = 0)
		verbose :  ``boolean`` (default True)
			print a line in console for every file read
		"""
		self.output_repos = output_repos
		self.iout = iout
		self.icpu = icpu
		self.part_read_fields = part_read_fields
		self.lbox = lbox

		self._ffkwargs = fortran_file_kwargs
		self.select_stars = select_stars
		self.select_dark_matter = select_dark_matter
		self.part_fname = filename_utils.amrlike_filename(
				"part", output_repos, iout, icpu, check_exists=True)
		self.verbose = verbose



	def read(self, read_lmax=None, fields_to_read=None):
		"""Read the particles data from the .part file and return a PointDataset
		containing the  user-defined fields
		Parameters
		----------
		read_lmax              : :  ``boolean`` (default None)
			level max to read
		fields_to_read          : : ``list of string`` (default None)
			fields to read like ["mass", "level", "epoch"]
		Returns
		-------
		dset : ``pymses.core.datasets.PointDataset``
			
		"""
		if fields_to_read != None:
			self.part_read_fields = fields_to_read
		keep_epoch_field_after_filter = True
		if (not self.select_stars or not self.select_dark_matter)\
			and not ("epoch" in self.part_read_fields):
			keep_epoch_field_after_filter = False
			self.part_read_fields.append("epoch")
		# Read the particles data from the particle file
		if self.verbose: print("Reading particles : %s" % self.part_fname)
		part_header, part_data = read_ramses_particle_file(
			self.part_fname, self.lbox)
		if "epoch" not in part_data:
			if ("epoch" in self.part_read_fields):
				self.part_read_fields.remove("epoch")
			if (not self.select_stars or not self.select_dark_matter):
				print("Warning : \"epoch\" field not found -> no particle selection is done here")
				self.select_stars = True
				self.select_dark_matter = True
		if (not self.select_stars or not self.select_dark_matter):
			# We have to filter stars or dm parts
			mask = numpy.ones(part_data["pos"].shape[0], 'bool')
			if not self.select_stars:
				mask = mask * (part_data["epoch"]==0) # select dm only
			if not self.select_dark_matter:
				mask = mask * (part_data["epoch"]!=0) # select stars only
			if part_data["pos"].shape[0] != 0:
				dset = PointDataset(part_data["pos"][mask]) # Dataset creation
				# Filling Dataset with the user-chosen fields
				if "vel" in self.part_read_fields:
					dset.add_vectors("vel", part_data["vel"][mask])
				if "mass" in self.part_read_fields:
					dset.add_scalars("mass", part_data["mass"][mask])
				if "id" in self.part_read_fields:
					dset.add_scalars("id", part_data["id"][mask])
				if "level" in self.part_read_fields:
					dset.add_scalars("level", part_data["level"][mask])
				if "epoch" in self.part_read_fields and keep_epoch_field_after_filter:
					dset.add_scalars("epoch", part_data["epoch"][mask])
				if "metal" in self.part_read_fields:
					dset.add_scalars("metal", part_data["metal"][mask])
				return dset
		# else:
		# We don't have to filter stars or dm parts
		dset = PointDataset(part_data["pos"]) # Dataset creation
		# Filling Dataset with the user-chosen fields
		if "vel" in self.part_read_fields:
			dset.add_vectors("vel", part_data["vel"])
		if "mass" in self.part_read_fields:
			dset.add_scalars("mass", part_data["mass"])
		if "id" in self.part_read_fields:
			dset.add_scalars("id", part_data["id"])
		if "level" in self.part_read_fields:
			dset.add_scalars("level", part_data["level"])
		if "epoch" in self.part_read_fields and keep_epoch_field_after_filter:
			dset.add_scalars("epoch", part_data["epoch"])
		if "metal" in self.part_read_fields:
			dset.add_scalars("metal", part_data["metal"])
			
		return dset

