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
import pymses.utils.point_utils as point_utils
import numpy


def test_meshgrid_2d():

	nx, ny = 10, 20
	xgrid, ygrid = numpy.mgrid[0:nx, 0:ny]
	pgrid = numpy.concatenate([xgrid, ygrid])
	plist = pgrid.reshape((2, -1)).transpose()

	xpts = numpy.arange(nx)
	ypts = numpy.arange(ny)
	grid = point_utils.meshgrid([xpts, ypts])

	assert numpy.all(grid == plist)


def test_meshgrid_3d():

	nx, ny, nz = 20, 30, 50
	xgrid, ygrid, zgrid = numpy.mgrid[0:nx, 0:ny, 0:nz]
	pgrid = numpy.concatenate([xgrid, ygrid, zgrid])
	plist = pgrid.reshape((3, -1)).transpose()

	xpts = numpy.arange(nx)
	ypts = numpy.arange(ny)
	zpts = numpy.arange(nz)
	grid = point_utils.meshgrid([xpts, ypts, zpts])

	assert numpy.all(grid == plist)


def test_meshgrid_4d():

	nx, ny, nz, nw = 20, 30, 50, 17
	xgrid, ygrid, zgrid, wgrid = numpy.mgrid[0:nx, 0:ny, 0:nz, 0:nw]
	pgrid = numpy.concatenate([xgrid, ygrid, zgrid, wgrid])
	plist = pgrid.reshape((4, -1)).transpose()

	xpts = numpy.arange(nx)
	ypts = numpy.arange(ny)
	zpts = numpy.arange(nz)
	wpts = numpy.arange(nw)
	grid = point_utils.meshgrid([xpts, ypts, zpts, wpts])

	assert numpy.all(grid == plist)


def test_meshgrid_indexing():
	nx, ny, nz = 20, 30, 50
	xpts = numpy.arange(nx)
	ypts = numpy.arange(ny)
	zpts = numpy.arange(nz)
	grid = point_utils.meshgrid([xpts, ypts, zpts])

	assert tuple(
			grid.reshape(nx, ny, nz, 3)[10, 14, 25, :] ) == (10., 14., 25.)

def test_compute_same_value_pixel_size_map():
	# TODO
	# !! This will evolve and change with parameters !!
	# map numbers generated with :
	# map = numpy.random.random((5,5))
	# map = numpy.array(map*5,'i')
	map = numpy.array(\
		[[2, 0, 3, 3, 0],
		 [4, 1, 4, 0, 3],
		 [4, 0, 3, 4, 3],
		 [1, 1, 3, 2, 2],
		 [3, 4, 1, 1, 3]], 'f')
	same_value_pixel_size_map_asserted_result = numpy.array(\
		[[0, 0, 2, 2, 1],
		 [1, 0, 1, 1, 2],
		 [1, 0, 1, 1, 2],
		 [3, 3, 1, 1, 1],
		 [0, 0, 3, 3, 0]], 'i')
	filtered_map_asserted_result = numpy.array(\
		[[ 2.        , 0.        , 2.0011166 , 1.79405243, 1.19373007],
		 [ 2.6296569 , 1.        , 2.2094049 , 2.40980212, 2.0884136 ],
		 [ 2.45708771, 0.        , 2.27524973, 2.70881434, 2.35359556],
		 [ 2.51044634, 2.41439667, 2.15545216, 2.39791002, 2.47302363],
		 [ 3.        , 4.        , 2.43638318, 2.42615966, 3.        ]], 'f')
	
	print("map :\n", map, map.min(),map.max())
	from pymses.utils.point_utils import adaptive_gaussian_blur, compute_same_value_pixel_size_map
	# same_value_pixel_size_map
	same_value_pixel_size_map = compute_same_value_pixel_size_map(map)
	print("same_value_pixel_size_map_asserted_result :\n", same_value_pixel_size_map_asserted_result,\
		same_value_pixel_size_map_asserted_result.min(), same_value_pixel_size_map_asserted_result.max())
	print("same_value_pixel_size_map :\n", same_value_pixel_size_map, same_value_pixel_size_map.min(),\
	same_value_pixel_size_map.max())
	#assert (same_value_pixel_size_map_asserted_result == same_value_pixel_size_map).all()
	
	# adaptive_gaussian_blur
	filtered_map = adaptive_gaussian_blur(map, same_value_pixel_size_map)
	print("filtered_map :\n", filtered_map, filtered_map.min(), filtered_map.max())
	print("filtered_map_asserted_result :\n", filtered_map_asserted_result,\
		filtered_map_asserted_result.min(), filtered_map_asserted_result.max())
	#assert (abs(filtered_map_asserted_result - filtered_map) < 10e-6).all()
	print("Test passed!")