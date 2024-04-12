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
//include <stdio.h>
//include <math.h>
//include "sample_point.h"

static void sample(double * grid_centers, int * son_indices,
		     double * scalars, int nscalars, double * point, int max_search_level,
		     int ndim, int twotondim, int *ilevel, int *igrid,
		     int *icell, double * extracted_data, int ngrids){
	int idim, ibit, son_ind, iscal;
	*igrid = 0;
	// Search down levels
	for (*ilevel = 1 ; *ilevel <= max_search_level ; (*ilevel)++){
		// Find the subcell
		*icell = 0;
		for (idim = 0 ; idim < ndim ; idim++){
			ibit = point[idim] > grid_centers[(*igrid)*ndim + idim];
			*icell += (ibit << idim);
		}
		son_ind = son_indices[(*igrid)*twotondim + *icell];
		// Is the subcell a leaf cell?
		if ((son_ind < 0) || (*ilevel == max_search_level)){
			// We're done searching: the subcell is the one we want
			break;
		}
		else{
			 //Go down the tree
			*igrid = son_ind;
		}
	}
	for (iscal = 0 ; iscal < nscalars ; iscal++){
		extracted_data[iscal] = scalars[iscal*twotondim*ngrids + (*igrid)*twotondim + (*icell)];
	}
}

static void sample_point_C(double * extracted_data, double * grid_centers,
			   int * son_indices, double * ccenter_array,
			   double * scalars, double * points,
			   int * max_search_levels, int ndim, int npoints,
			   int interpolation, int nscalars, int ngrids){
	int ipoint, idim, level, igrid, icell, ilevel, iscal, i;
	double *x0y0z0 = malloc(nscalars * sizeof(double));
	double *x0y0z1 = malloc(nscalars * sizeof(double));
	double *x0y1z0 = malloc(nscalars * sizeof(double));
	double *x0y1z1 = malloc(nscalars * sizeof(double));
	double *x1y0z0 = malloc(nscalars * sizeof(double));
	double *x1y0z1 = malloc(nscalars * sizeof(double));
	double *x1y1z0 = malloc(nscalars * sizeof(double));
	double *x1y1z1 = malloc(nscalars * sizeof(double));
	double cell_size, x, y, z;
	double * point = malloc(ndim * sizeof(double));
	double * real_point = malloc(ndim * sizeof(double));
	double *cc;
	//printf("Start sampling points on AMR array...\n");
	int twotondim = 1 << ndim;
	int twotondimMndim = twotondim * ndim;
	int max_search_level;
	//for (iscal = 0 ; iscal < nscalars*8*9 ; iscal++){
	//	printf(" %e\n",scalars[iscal]);
	//}
	
	// Loop over points
	for (ipoint = 0 ; ipoint < npoints ; ipoint++){
		for (idim = 0 ; idim < ndim ; idim++){
			point[idim] = points[ipoint*ndim + idim];
			real_point[idim] = point[idim];
		}
		sample(grid_centers, son_indices, scalars, nscalars,
			point, max_search_levels[ipoint],
			ndim, twotondim, &level, &igrid, &icell, x0y0z0, ngrids);
		if (interpolation){
			max_search_level = level;
			// loop with 2 iterations :
			// normally all neighbour cells has the same level
			// and we skip the second iteration
			// but if a lower level is found : we go into
			// the second iteration with a reduced levelmax
			// there might be at most 2 levels of difference
			// (moving in diagonal by a lenght < cell_size )
			// so we need at worst 3 iterations
			for (i = 0 ; i < 3 ; i++){
				//printf(" %i ",max_search_levels[ipoint]);
				//printf(" %i ",level);
				// we compute the cell_size for this level :
				cell_size = 1;
				for (ilevel = 0 ; ilevel < level; ilevel++){
					cell_size *= .5;
				}
				// then we correct the difference between cell centered
				// values and cell corner values : we just have to
				// substract cell_size / 2 to the real (original ) point
				// coordinates
				for (idim = 0 ; idim < ndim ; idim++){
					point[idim] = real_point[idim] - cell_size * .5;
				}
				sample(grid_centers, son_indices, scalars, nscalars,
					point, max_search_level, ndim, twotondim,
					&level, &igrid, &icell, x0y0z0, ngrids);
				//printf(" %i ",i);
				cc = &ccenter_array[igrid*twotondimMndim + icell*ndim];
				if (level < max_search_level){
					// a lower level cell has been
					// found, so go to the next iteration
					// using this new levelmax
					max_search_level = level;
					continue;
				}
				// we compute first interpolation factors:
				x = (point[0] - cc[0])/cell_size + .5;
				y = (point[1] - cc[1])/cell_size + .5;
				if (ndim ==3) z = (point[2] - cc[2])/cell_size + .5;
				// then we can use again the local variable "point"
				if (ndim ==3){
					point[0] = cc[0];
					point[1] = cc[1];
					point[2] = cc[2] + cell_size;
					sample(grid_centers, son_indices, scalars, nscalars,
						point, max_search_level,
						ndim, twotondim, &level, &igrid, &icell, x0y0z1, ngrids);
					if (level < max_search_level){
						// a lower level neighbor cell has been
						// found, so go to the next iteration
						// using this new levelmax
						max_search_level = level;
						continue;
					}
				}
				//point[0] = cc[0];
				point[1] = cc[1] + cell_size;
				if (ndim ==3) point[2] = cc[2];
				sample(grid_centers, son_indices, scalars, nscalars,
					point, max_search_level,
					ndim, twotondim, &level, &igrid, &icell, x0y1z0, ngrids);
				if (level < max_search_level){
					// a lower level neighbor cell has been
					// found, so go to the next iteration
					// using this new levelmax
					max_search_level = level;
					continue;
				}
				if (ndim ==3){
					//point[0] = cc[0];
					//point[1] = cc[1] + cell_size;
					point[2] = cc[2] + cell_size;
					sample(grid_centers, son_indices, scalars, nscalars,
						point, max_search_level,
						ndim, twotondim, &level, &igrid, &icell, x0y1z1, ngrids);
					if (level < max_search_level){
						// a lower level neighbor cell has been
						// found, so go to the next iteration
						// using this new levelmax
						max_search_level = level;
						continue;
					}
				}
				point[0] = cc[0] + cell_size;
				point[1] = cc[1];
				if (ndim ==3) point[2] = cc[2];
				sample(grid_centers, son_indices, scalars, nscalars,
					point, max_search_level,
					ndim, twotondim, &level, &igrid, &icell, x1y0z0, ngrids);
				if (level < max_search_level){
					// a lower level neighbor cell has been
					// found, so go to the next iteration
					// using this new levelmax
					max_search_level = level;
					continue;
				}
				if (ndim ==3){
					//point[0] = cc[0] + cell_size;
					//point[1] = cc[1];
					point[2] = cc[2] + cell_size;
					sample(grid_centers, son_indices, scalars, nscalars,
						point, max_search_level,
						ndim, twotondim, &level, &igrid, &icell, x1y0z1, ngrids);
					if (level < max_search_level){
						// a lower level neighbor cell has been
						// found, so go to the next iteration
						// using this new levelmax
						max_search_level = level;
						continue;
					}
				}
				//point[0] = cc[0] + cell_size;
				point[1] = cc[1] + cell_size;
				if (ndim ==3) point[2] = cc[2];
				sample(grid_centers, son_indices, scalars, nscalars,
					point, max_search_level,
					ndim, twotondim, &level, &igrid, &icell, x1y1z0, ngrids);
				if (level < max_search_level){
					// a lower level neighbor cell has been
					// found, so go to the next iteration
					// using this new levelmax
					max_search_level = level;
					continue;
				}
				if (ndim ==3){
					//point[0] = cc[0] + cell_size;
					//point[1] = cc[1] + cell_size;
					point[2] = cc[2] + cell_size;
					sample(grid_centers, son_indices, scalars, nscalars,
						point, max_search_level,
						ndim, twotondim, &level, &igrid, &icell, x1y1z1, ngrids);
					if (level < max_search_level){
						// a lower level neighbor cell has been
						// found, so go to the next iteration
						// using this new levelmax
						max_search_level = level;
						continue;
					}
				}
				// no lower level neighbor cell has been
				// found, so we don't need to go into the 
				// next iteration of the interpolate loop
				break;
			}
			for (iscal = 0 ; iscal < nscalars ; iscal++){
				if (ndim ==3) {
					extracted_data[iscal*npoints + ipoint] = (1-x)*(1-y)*(1-z)*x0y0z0[iscal] +
									(1-x)*(1-y)*z*x0y0z1[iscal] +
							(1-x)*y*(1-z)*x0y1z0[iscal] +(1-x)*y*z*x0y1z1[iscal] +
							x*(1-y)*(1-z)*x1y0z0[iscal] + x*(1-y)*z*x1y0z1[iscal] +
							x*y*(1-z)*x1y1z0[iscal] + x*y*z*x1y1z1[iscal];
				}
				else {
					extracted_data[iscal*npoints + ipoint] = (1-x)*(1-y)*x0y0z0[iscal] +
							(1-x)*y*x0y1z0[iscal] +
							x*(1-y)*x1y0z0[iscal] +
							x*y*x1y1z0[iscal];
				}
			}
		}
		else{
			for (iscal = 0 ; iscal < nscalars ; iscal++){
				extracted_data[iscal*npoints + ipoint] = x0y0z0[iscal];
				//printf(" %e\n",extracted_data[ipoint*nscalars + iscal]);
				//printf(" %i ",igrid);
			}
		}
	}
	free(x0y0z0);
	free(x0y0z1);
	free(x0y1z0);
	free(x0y1z1);
	free(x1y0z0);
	free(x1y0z1);
	free(x1y1z0);
	free(x1y1z1);
	free(point);

}