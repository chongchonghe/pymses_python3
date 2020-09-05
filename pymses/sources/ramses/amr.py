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
amr.py -- flexible raw RAMSES AMR file reading
"""

from . import _read_ramses

def read_ramses_amr_file(amr_filename, max_read_level=None):#{{{
	""" Reads a RAMSES .amr file into memory

	Arguments:

		amr_filename -- filename of the .amr file

		max_read_level -- read all levels <= max_read_level (default: read all)

	Returns: (header_dict, amr_struct_dict)
	"""

	if max_read_level is None:
		max_read_level = 99

	out = _read_ramses.read_amr(amr_filename, max_read_level)
	return out
#}}}
