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
""" filename_utils.py -- utility functions for RAMSES outputs filename
management
"""
import os
from functools import wraps
import re

def wrap_check_exists(func):
	@wraps(func)
	def wrapped_func(*args, **kwargs):
		check = kwargs.pop("check_exists", True)
		path = func(*args, **kwargs)

		if check and not os.path.exists(path):
			raise ValueError("path %s does not exist" % path)

		return path
	return wrapped_func


@wrap_check_exists
def output_path(output_repos, iout):
	"Assembles a RAMSES output path from an output repository and output number"
	return os.path.join(output_repos, "output_%05i" % iout)


@wrap_check_exists
def info_filename(output_repos, iout):
	"Assembles a RAMSES .info file name"

	path = output_path(output_repos, iout)
	return os.path.join(path, "info_%05i.txt" % iout)

@wrap_check_exists
def sink_filename(output_repos, iout):
	"Assembles a .sink file name"

	path = output_path(output_repos, iout)
	return os.path.join(path, "sink_%05i.csv" % iout)

@wrap_check_exists
def amrlike_filename(ftype, output_repos, iout, icpu):
	"""Assembles a RAMSES per-CPU file name.

	Parameters
	----------
	ftype : ``string``
		the file type: "amr", "hydro", "grav", "part", etc
	output_repos : ``string``
		the directory path containing the "output_*" directories
	iout : ``int``
		the output number
	icpu : ``int``
		the CPU number

	Returns
	-------
	filename : ``string``
		the complete path of the amr-like cpu file

	"""
	path = output_path(output_repos, iout)

	filename = os.path.join(path,
			"%s_%05i.out%05i" % (ftype, iout, icpu))

	return filename

def output_files_dict(output_repos, iout):
	"""Lists all the files of a given output path by type in a dictionary
	"""

	path = output_path(output_repos, iout)
	output_flist = os.listdir(path)
	output_dict = {}

	for outfile in output_flist:
		ftype = outfile.split("_")[0]

		if ftype not in output_dict:
			output_dict[ftype] = []

		output_dict[ftype].append(outfile)

	# Sort the lists
	for lst in output_dict.values():
		lst.sort()

	return output_dict

def search_valid_outputs(out_dir):
	r"""
	Computes the ``int`` ``list`` of output number available in a given `out_dir`
	RAMSES outputs directory

	Parameters
	----------

	out_dir : ``string``
		path of the directory containing all RAMSES outputs

	Returns
	-------
	ilist : ``list``
		sorted number list of the available outputs

	"""
	ilist = []
	outdir_regexp = re.compile("output_[0-9]{5}")
	iout_regexp = re.compile("[0-9]{5}")
	ls = os.listdir(out_dir)
	for file in ls:
		res = outdir_regexp.findall(file)
		if len(res) > 0:
			if res[0]==file:
				ilist.append(int(iout_regexp.findall(file)[0]))

	ilist.sort()
	return ilist

__all__ = ["search_valid_outputs",
		   "output_path", "output_files_dict",
		   "info_filename", "amrlike_filename"]
