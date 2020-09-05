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
#ifndef _SAMPLE_POINT
#define _SAMPLE_POINT
#endif

static void sample(double * grid_centers, int * son_indices,
		     double * scalars, int nscalars, double * point, int max_search_level,
		     int ndim, int twotondim, int *ilevel, int *igrid,
		     int *icell, double * extracted_data, int ngrids);

static void sample_point_C(double * extracted_data, double * grid_centers,
			   int * son_indices, double * ccenter_array,
			   double * scalars, double * points,
			   int * max_search_level, int ndim, int npoints,
			   int interpolation, int nscalars, int ngrids);
