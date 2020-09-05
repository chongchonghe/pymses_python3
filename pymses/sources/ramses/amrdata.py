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
amrdata.py -- flexible raw RAMSES AMR data file reading
"""

from . import _read_ramses

def read_ramses_amr_data_file(filetype, data_filename, amr, ivars_to_read):
	r"""
	Reads a RAMSES AMR data file into memory (.hydro, .grav, ...)

	Parameters
	----------
	filetype      : ``string``
		filetype of the data file to read ("grav", "hydro", ...)
	data_filename : ``string``
		filename of the data file
	amr           : AMR structure
		the AMR structure, as output by read_ramses_amr_file.
	ivars_to_read : ``list`` of ``int``
		list of variable ids to read

	Returns
	-------
	d : ``dict``
		a dictionnary of data arrays, indexed by ivar

	"""

	amrhdr, amrstruct = amr
	
	datadict = _read_ramses.read_cells(filetype, data_filename,
			amrstruct["ngrids"], amrstruct["readlmax"], ivars_to_read)

	return datadict
