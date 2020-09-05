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
info.py -- RAMSES .info reading routine
"""

import re
import numpy
import pymses.utils.constants as C

def read_ramses_info_file(info_filename):
	""" Reads a RAMSES .info file, returning a dictionary of parameter values
	"""

	info_dict = {}
	info_fileobj = open(info_filename, 'r')
	
	re_split = re.compile("\s*=\s*")
	re_emptyline = re.compile("^\s*$")
	re_header_boundkeys = re.compile("^\s*DOMAIN\s*ind_min\s*ind_max\s*$")
	re_head_bkey = re.compile("0\.")
	re_tail_bkey = re.compile("E\+[0-9]+")

	bound_table = None
	par_dict = {}
	is_param = True

	# TODO: some code cleanup down there
	for line in info_fileobj.readlines():
		if re_emptyline.match(line):
			continue
		if re_header_boundkeys.match(line):
			is_param = False
			continue
		params = re_split.split(line)
		if is_param:
			par_name, par_val = params
			par_dict[par_name] = par_val
			if par_name == "ordering type":
				if par_val.strip() != "hilbert":
					break
				else:
					re_split = re.compile("\s*")
					bound_table = []
					par_dict[par_name] = par_val.strip()
		else:
			bound_table.append([float(params[2]), float(params[3])])
			params2 = re_head_bkey.split(params[2])
			params3 = re_tail_bkey.split(params2[1])
			nd = len(params3[0])

	info_dict["ncpu"] = int(par_dict["ncpu"])
	info_dict["ndim"] = int(par_dict["ndim"])
	info_dict["levelmin"] = int(par_dict["levelmin"])
	info_dict["levelmax"] = int(par_dict["levelmax"])
	info_dict["ngridmax"] = int(par_dict["ngridmax"])
	info_dict["nstep_coarse"] = int(par_dict["nstep_coarse"])
	info_dict["boxlen"] = float(par_dict["boxlen"])
	info_dict["time"] = float(par_dict["time"])
	info_dict["aexp"] = float(par_dict["aexp"])
	info_dict["H0"] = float(par_dict["H0"])
	info_dict["omega_m"] = float(par_dict["omega_m"])
	info_dict["omega_l"] = float(par_dict["omega_l"])
	info_dict["omega_k"] = float(par_dict["omega_k"])
	info_dict["omega_b"] = float(par_dict["omega_b"])
	info_dict["ordering"] = par_dict["ordering type"]
	info_dict["unit_length"] = float(par_dict["unit_l"])*C.cm
	info_dict["unit_density"] = float(par_dict["unit_d"])*C.g/C.cm**3
	info_dict["unit_time"] = float(par_dict["unit_t"])*C.s
	info_dict["unit_velocity"] = info_dict["unit_length"] / info_dict["unit_time"]
	info_dict["unit_pressure"] = info_dict["unit_density"] \
			* info_dict["unit_velocity"]**2
	info_dict["unit_temperature"] = info_dict["unit_velocity"]**2 \
			* C.mH / C.kB
	info_dict["unit_mag"] = (4*numpy.pi*float(par_dict["unit_d"]) \
			*(float(par_dict["unit_l"])/float(par_dict["unit_t"]))**2)**(1./2)*C.Gauss
	
	# Boxlen renormalisation : used only for dark matter particles,
	# so we do this just before computing the mass unit
	info_dict["unit_length"] = info_dict["unit_length"] * info_dict["boxlen"]
	info_dict["unit_mass"] = info_dict["unit_density"] \
			* info_dict["unit_length"]**3
	
	
	if info_dict["ordering"] == "hilbert":
		keys = numpy.zeros(info_dict["ncpu"]+1)
		bound_table = numpy.asarray(bound_table)
		keys[0] = bound_table[0,0]
		keys[1:] = bound_table[:,1]
		info_dict["dom_decomp_Hilbert_keys"] = keys
	else:
		info_dict["dom_decomp"] = None

	info_fileobj.close()
	return info_dict

