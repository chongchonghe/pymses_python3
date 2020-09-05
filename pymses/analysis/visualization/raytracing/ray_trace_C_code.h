/******************************************************************************************
* License:                                                                                *
*   Copyright (C) 2011 Thomas GUILLET, Damien CHAPON, Marc LABADENS. All Rights Reserved. *
*                                                                                         *
*   This file is part of PyMSES.                                                          *
*                                                                                         *
*   PyMSES is free software: you can redistribute it and/or modify                        *
*   it under the terms of the GNU General Public License as published by                  *
*   the Free Software Foundation, either version 3 of the License, or                     *
*   (at your option) any later version.                                                   *
*                                                                                         *
*   PyMSES is distributed in the hope that it will be useful,                             *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of                        *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                         *
*   GNU General Public License for more details.                                          *
*                                                                                         *
*   You should have received a copy of the GNU General Public License                     *
*   along with PyMSES.  If not, see <http://www.gnu.org/licenses/>.                       *
******************************************************************************************/
#ifndef _RAY_TRACE_C_CODE_H
#define _RAY_TRACE_C_CODE_H
#endif

#include<math.h> 

static void ray_trace_C(double * maps, double * ray_length, double * ray_origins, double * ray_vects,
	double * cell_centers, int * sons, int * g_levels, double * scal_data,
	int * igrids, char * icells, long * active_mask, double * ray_depth_max,
	long * param_info_int);

static void ray_trace_octree_C(double * I, double * ray_origins, double * ray_vects,
	double * grid_centers, double * cell_centers, int * sons, int * cell_levels,
	double * scalar_data, int * neighbors, double * ray_lengths,
	long * param_info_int, double gamma, double * vlines, double * wlines,
	double * rlines, double * glines, double * blines, double * alines);

//cdef extern from "test_C_code.h":
//	double test_M(double x)
//	
//static double test_M(double a);
//static double test_M(double a) {
//	printf("test_M!\n");
//	return a * a;
//}
