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
:mod:`pymses.sources.ramses.output` --- RAMSES output package
-------------------------------------------------------------

"""
from . import info, filename_utils, octree, particles, sources
import os.path, re
from .hilbert import HilbertDomainDecomp
from numpy import concatenate, expand_dims, fromfile

class Scalar(object):#{{{
	def __init__(self, name, ivar):
		self.ivar = int(ivar)
		self.ivars = [self.ivar]
		self.name = name

	def gather(self, data, dset):
		dset.add_scalars(self.name, data[self.ivar])
#}}}

class Vector(object):#{{{
	def __init__(self, name, ivars):
		# Ensure integer ivars
		self.ivars = [int(i) for i in ivars]
		self.name = name

	def gather(self, data, dset):
		vect = concatenate([expand_dims(data[ivar], axis=2)	for ivar in self.ivars], axis=2)
		dset.add_vectors(self.name, vect)
#}}}

class RamsesOutput(object):#{{{

	amr_field_descrs_by_file = \
		{"2D": {"hydro" : [ Scalar("rho", 0), Vector("vel", [1, 2]), Scalar("P", 3) ],
			"grav"  : [ Vector("g", [0, 1]) ]
			},
			"3D": {"hydro" : [ Scalar("rho", 0), Vector("vel", [1, 2, 3]), Scalar("P", 4) ],
			     "grav"  : [ Vector("g", [0, 1, 2]) ]
			}
		}
	r"""
	RAMSES output class

	Parameters
	----------
	output_repos : ``string``
		path of the RAMSES output repository, containing all the simulation outputs
	iout         : ``int``
		output number

	Examples
	--------

		>>> ro = RamsesOutput() # select current directory first output
		>>> ro = RamsesOutput("/data/Aquarius/outputs", 193)
	
	"""
	def __init__(self, output_repos=None, iout=None, verbose=True):#{{{
		r"""
		Build a RamsesOutput object from an output repository path and the number of the output.
		Select current directory first output by default
		Parameters
		----------
		output_repos : ``string`` (default None)
			path to output repository
		iout : ``int`` (default None)
			output number
		verbose : ``boolean`` (default True)
			print in console the name of every file readed
		"""
		if output_repos == None:
			output_repos = "./"
		if iout == None:
			ilist = filename_utils.search_valid_outputs(output_repos)
			ok = (len(ilist) > 0)
			if not ok :
				# try to see if up directory is not already a good directory
				ilist = filename_utils.search_valid_outputs("."+output_repos)
				ok = (len(ilist) > 0)
				if ok:
					# Find output number iout
					outName = os.path.basename(os.path.realpath(output_repos))
					iout_regexp = re.compile("[0-9]{5}")
					iout = int(iout_regexp.findall(outName)[0])
					output_repos = os.path.realpath("."+output_repos)
			else :
				output_repos = os.path.realpath(output_repos)
				iout = ilist[0]
		self.output_repos = output_repos
		self.iout = iout
		self.verbose = verbose
		self.sink = None
		self.mhd=None
		
		# Read the .info file
		self.info = info.read_ramses_info_file(
			filename_utils.info_filename(output_repos, iout) )
		self.ndim = self.info["ndim"]
		self.ncpu = self.info["ncpu"]
		self.cpu_list = list(range(1,self.ncpu+1))
		if ((self.info["ordering"] == "hilbert")):
			keys = self.info["dom_decomp_Hilbert_keys"]
			print("Computing hilbert minimal domain description for output",iout,"...")
			self.info["dom_decomp"] = HilbertDomainDecomp(self.ndim, \
					keys[:-1], keys[1:], \
					(self.info["levelmin"], self.info["levelmax"]) )
			print("Done !")
		
		# If it is an MHD simulation, update the field_descr adding magnetic field
		hname = filename_utils.amrlike_filename('hydro', self.output_repos, self.iout, self.ncpu) 
		var_number = fromfile(hname, count=5, dtype='i4')[4]
		if (var_number == 11 or var_number == 14):
			self.amr_field_descrs_by_file = \
	{"2D":	{"hydro" : [ Scalar("rho", 0), Vector("vel", [1, 2, 3]), 
		Vector("Bl", [4, 5, 6]), Vector("Br", [7, 8, 9]), 
		Scalar("P", 10) ],
		"grav"  : [ Vector("g", [0, 1, 2]) ]
		},
	"3D": 	{"hydro" : [ Scalar("rho", 0), Vector("vel", [1, 2, 3]), 
		Vector("Bl", [4, 5, 6]), Vector("Br", [7, 8, 9]), 
		Scalar("P", 10) ],
		"grav"  : [ Vector("g", [0, 1, 2]) ]
		}
	}
			print(var_number, "variables found - Classic MHD simulation amr_field_descrs_by_file loaded !")
			self.mhd=True
		# Update variable description if the appropriate file is found :
		try:
			try:
				#f=open(filename_utils.output_path(output_repos, iout)+"/pymses_field_descrs.py")
                                f=open(output_repos+"/pymses_field_descrs.py")
			except Exception:
				f=open("./pymses_field_descrs.py")
			code = f.read()
			exec(code)
		except Exception as e:
			print("Warning :",var_number,"variables found - Using default value for pymses_field_descrs.py because ", e)

		# List all the available files
		self.output_files = filename_utils.output_files_dict(output_repos, iout)

		# Retrieve the domain decomposition
		self.dom_decomp = self.info["dom_decomp"]
#}}}

	def amr_source(self, amr_read_fields, cpu_list=None):#{{{
		r"""
		Return a RamsesAmrSource, able to read a set of amr fields


		Parameters
		----------
		amr_read_fields : ``list`` of ``strings``
			list of AMR data fields that needed to be read
		cpu_list : ``list`` of ``int`` (default None)
			If specified, restricts the cpu list to this list (for
			faster initialization). Default : cpu_list = range(1,ncpu+1)
		
		Returns
		-------
		ramses_amr_source : :class:`~pymses.sources.ramses.sources.RamsesAmrSource`
			RAMSES AMR data source
		
		"""
		nD = "%iD"%self.ndim
		# Transpose the fields_by_file dictionary to allow field -> files mapping
		fields_by_name = dict(
			[ ( field_descr.name, (amr_file_type, field_descr) )
				for amr_file_type in self.amr_field_descrs_by_file[nD]
					for field_descr in self.amr_field_descrs_by_file[nD][amr_file_type] ])
		# Sanity checks
		all_fields = list(fields_by_name.keys())
		for field_name in amr_read_fields:
			assert (field_name in all_fields), "Error : unknown '%s' amr field !"%field_name

		# Gather the descriptors of the AMR fields we need to read, grouped by file
		req_descrs_by_file = {}
		for field_name in amr_read_fields:
			amr_file_type, field_descr = fields_by_name[field_name]
			if amr_file_type in req_descrs_by_file:
				req_descrs_by_file[amr_file_type].append(field_descr)
			else:
				req_descrs_by_file[amr_file_type] = [field_descr]

		ivars_descrs_by_file = {}
		for amr_file_type, descrs in req_descrs_by_file.items():
			# Gather the ivars, uniquify and sort them
			ivars_to_read = []
			for descr in descrs:
				ivars_to_read += descr.ivars
			ivars_descrs_by_file[amr_file_type] = (sorted(list(set(ivars_to_read))), descrs)
		if cpu_list != None:
			data_list = cpu_list
		else :
			data_list = self.cpu_list
		reader_list = [
				octree.RamsesOctreeReader(self.output_repos, self.iout, 
					icpu, ivars_descrs_by_file, self.verbose)
				for icpu in data_list ]

		return sources.RamsesAmrSource(reader_list, self.dom_decomp, data_list, self.ndim, amr_read_fields)
	#}}}


	def particle_source(self, part_read_fields, cpu_list=None, select_stars=True,
			select_dark_matter=True):#{{{
		r"""
		Return a RamsesParticleSource, able to read a set of user-defined particle data fields.


		Parameters
		----------
		part_read_fields : ``list`` of ``strings``
			list of particle data fields that need to be read
		cpu_list : ``list`` of ``int`` (default None)
			If specified, restricts the cpu list to this list (for
			faster initialization). Default : cpu_list = range(1,ncpu+1)
		select_stars :  ``boolean`` (default True)
			if True :  select and read STARS particules
				(with "epoch" field != 0)
		select_dark_matter :  ``boolean`` (default True)
			if True : select and read only DARK MATTER particules
				(with "epoch" field = 0)
		
		Returns
		-------
		ramses_part_source : :class:`~pymses.sources.ramses.sources.RamsesParticleSource`
			RAMSES particle data source
		
		"""
		if cpu_list != None:
			data_list = cpu_list
		else :
			data_list = self.cpu_list
		reader_list = [
				particles.RamsesParticleReader(self.output_repos, self.iout, 
					icpu, part_read_fields, self.info["boxlen"],
					select_stars=select_stars, select_dark_matter=select_dark_matter,
					verbose=self.verbose)
				for icpu in self.cpu_list ]

		return sources.RamsesParticleSource(reader_list, self.dom_decomp, data_list, self.ndim, part_read_fields)
	#}}}

#}}}

__all__ = ["RamsesOutput", "Scalar", "Vector"]
