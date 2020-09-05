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
import numpy
from pymses.filters import *
from pymses.analysis.visualization import *

def find_galaxy_axis(points_dset_source, camera, nbSample=2000):
	r"""
	If a galaxy disk is centered in the camera box, this function should
	return a galaxy disk othogonal axis.
	
	from pymses.analysis import find_galaxy_axis
	
	Parameters
	----------
	points_dset_source : :ref:`PointDataSource`
		fields "rho" and "size" needed
		
	camera : pymses camera, the galaxy's center has to fit the camera's center 
	
	nbSample : int (default=2000)
		number of max massive points to use to compute the axis through cross product
	"""
	filtered_points_dset_source = RegionFilter(camera.get_bounding_box(), points_dset_source)
	filtered_points_dset = filtered_points_dset_source.flatten() # multiprocessing data reading and filtering
	region_filtered_mesh_mass = filtered_points_dset.fields["rho"]*(filtered_points_dset.fields["size"]**3)
	argsort = numpy.argsort(region_filtered_mesh_mass)
	center=camera.center
	nbSample = min(nbSample, argsort.size/2-1)
	result_vect = numpy.array([0.,0.,0.])
	for i in range (nbSample):
		index1 = argsort[-2*i]
		index2 = argsort[-2*i-1]
		vect1 = filtered_points_dset.points[index1] - center
		vect2 = filtered_points_dset.points[index2] - center
		vect = numpy.cross(vect1, vect2)
		sign = numpy.dot(vect, [0.,0.,1.])
		if sign < 0 :
			vect = - vect
		result_vect = result_vect + vect * (region_filtered_mesh_mass[index1] +
						    region_filtered_mesh_mass[index2]) * \
						numpy.linalg.norm((vect1-vect2),2)
	result_vect = result_vect/numpy.linalg.norm(result_vect,2)
	return result_vect

def find_center_of_mass(points_dset_source, camera, nbSample=2000):
	r"""
	Find the center of mass in the camera box
	
	Parameters
	----------
	points_dset_source : :ref:`PointDataSource`
		fields "rho" and "size" needed
		
	camera : pymses camera box definition restriction
	
	nbSample : int (default=2000)
		not working yet : may speed up if random sampling ?
	"""
	filtered_points_dset_source = RegionFilter(camera.get_bounding_box(), points_dset_source)
	filtered_points_dset = filtered_points_dset_source.flatten() # multiprocessing data reading and filtering
	d = filtered_points_dset.fields["rho"]*(filtered_points_dset.fields["size"]**3)
	mass=numpy.sum(d)
	cm=numpy.sum(d[:,numpy.newaxis]*filtered_points_dset.points,axis=0)/mass
	return cm

def find_los(points_dset_source, camera, nbSample=2000):
	r"""
	Find the line of sight axis which is along the angular momentum of the gas inside the camera box
	
	Parameters
	----------
	points_dset_source : :ref:`PointDataSource`
		fields "vel", "rho" and "size" needed
		
	camera : pymses camera box definition restriction
	
	nbSample : int (default=2000)
		not working yet : may speed up if random sampling ?
	"""
	filtered_points_dset_source = RegionFilter(camera.get_bounding_box(), points_dset_source)
	filtered_points_dset = filtered_points_dset_source.flatten() # multiprocessing data reading and filtering
	d = filtered_points_dset.fields["rho"]*(filtered_points_dset.fields["size"]**3)
	v=filtered_points_dset["vel"]
	JJ=numpy.zeros_like(v)
	p=d[:,numpy.newaxis]*v
	JJ[:]=numpy.cross((filtered_points_dset.points[:]-camera.center),p[:])
	J=numpy.sum(JJ,axis=0)
	result_vect = J/sum(J**2)
	result_vect = result_vect/numpy.linalg.norm(result_vect,2)
	return result_vect

__all__ = ["find_galaxy_axis", "find_center_of_mass", "find_los"]