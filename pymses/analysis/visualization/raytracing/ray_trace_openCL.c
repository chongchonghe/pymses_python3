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
${precision_pragma}
__kernel
		void rt(__global ${precision} *I, __global ${precision} *ray_origins,
		__global const ${precision} *ray_vects, __global const ${precision} *grid_centers,
		__global const ${precision} *cell_centers, __global const int *sons,
		__global const uint *cell_levels, __global const ${precision} *scalar_data,
		__global const uint *neighbors, __global const ${precision} *ray_lengths,
		__global const ${precision} *dx,
		uint ndim, uint max_read_level, uint nfunc, uint max_op, uint ngrids, uint rgb, uint nlines,
		${precision} gamma, __global const ${precision} *vlines,
		__global const ${precision} *wlines, __global const ${precision} *rlines,
		__global const ${precision} *glines, __global const ${precision} *blines,
		__global const ${precision} *alines
		){
	uint twotondim = 1 << ndim;
	uint twoMdim = 2 * ndim;
	uint twotondimMndim = twotondim * ndim;
	uint twotondimMngrids = twotondim * ngrids;
	
	// integer variables
	uint igridn, ileveln, step;
	uint idim2, iface;
	int sgn_face, isgn, indn, ind_son;
	uint idz, ndz, iem;
	uint irgb, ifunc,test_ok;
	uint idim, ibit, ilevel, ind;
	uint error = 0;
	uint iray = get_global_id(0);
	// ${precision} variables
	${precision} v, vl, t_0, t_1, tmin, tmax;
	${precision} scalar, scalar2, cc, scalarUp, scalarUp2;
	${precision} ray_vect_idim, ray_orig_idim, fc_idim, fcoord, fcoordn;
	${precision} zbefore, z, zin, zout, z_intersect;
	${precision} dx_min, dz, scalar_loc, zvar, dx_loc, dx_2;
	${precision} zmax = ray_lengths[iray];
	${precision} em, rem, gem, bem, alpha, gauss_value, absor, width, x;
	// Integration of the ray intensities (RGBA) between 0 (ray origin) and zmax
	
	//######## Project ray origin inside the simulation box if it was outside #####
	if (ray_origins[iray * ndim + 0] < 0 || ray_origins[iray * ndim + 0] > 1 ||
	ray_origins[iray * ndim + 1] < 0 || ray_origins[iray * ndim + 1] > 1 ||
	ray_origins[iray * ndim + 2] < 0 || ray_origins[iray * ndim + 2] > 1){
		// ray_origin is outside the simulation box
		t_0 = (0 - ray_origins[iray * ndim + 0]) / ray_vects[iray * ndim + 0];
		t_1 = (1 - ray_origins[iray * ndim + 0]) / ray_vects[iray * ndim + 0];
		tmin = min(t_0, t_1);
		tmax = max(t_0, t_1);
		t_0 = (0 - ray_origins[iray * ndim + 1]) / ray_vects[iray * ndim + 1];
		t_1 = (1 - ray_origins[iray * ndim + 1]) / ray_vects[iray * ndim + 1];
		tmin = max(tmin, min(t_0, t_1));
		tmax = min(tmax, max(t_0, t_1));
		t_0 = (0 - ray_origins[iray * ndim + 2]) / ray_vects[iray * ndim + 2];
		t_1 = (1 - ray_origins[iray * ndim + 2]) / ray_vects[iray * ndim + 2];
		tmin = max(tmin, min(t_0, t_1));
		tmax = min(tmax, max(t_0, t_1));
		if ((tmin < tmax) && (0 < tmin)){
			for (idim=0 ; idim < ndim ; idim++)
				ray_origins[iray * ndim + idim] += tmin * ray_vects[iray * ndim + idim] * 1.001;
		}
		else{
			// Projection not possible : we skip this ray !
			error = 1;
		}
	}
	if (!error)
	{
		//#####/ Tree-search in octree to find where is the current ray origin #####/
		uint igrid = 0;
		int igrid_son = 0;
		for (ilevel = 0 ; ilevel < max_read_level ; ilevel++){
			if (ilevel != 0) igrid = igrid_son;
			// Find son index in octree
			ind = 0;
			for (idim=0 ; idim < ndim ; idim++){
				ibit = (ray_origins[iray * ndim + idim] > grid_centers[igrid * ndim + idim]);
				ind += (ibit << idim);
			}
			igrid_son = sons[igrid * twotondim + ind];
			if (igrid_son < 0) break;
		}
		//I[iray*3+0]=ind;
		//I[iray*3+1]=igrid_son;
		//I[iray*3+2]=ilevel;
		//####################################################
		// Now the ray_origin is in a leaf-cell of the current grid
		zbefore = 0;
		z=0; // Set the position along ray to 0 (ray origin)
		scalar = scalar_data[igrid * twotondim + ind];
		scalarUp = scalar_data[twotondimMngrids + igrid * twotondim + ind];
		//############  Initialisation : first cell ray/face intersection computation ###########/
		dx_loc = dx[ilevel];
		dx_2 = dx[ilevel+1]; // dx_loc/2.0
		// Finding the exit intersection of the ray and the current cell
		test_ok = 0;
		for (idim=0 ; idim < ndim ; idim++){
			ray_vect_idim = ray_vects[iray * ndim + idim];
			ray_orig_idim = ray_origins[iray * ndim + idim];
			if ((ray_vect_idim == 0)){ // ray is parallel to the faces perpendicular to dimension idim.
				// There cannot be any intersection with the faces.
				continue;
			}
			else{ if (ray_vect_idim > 0)
					sgn_face = 1;
				else
					sgn_face = -1;
				// Coordinate of the cell face center along dimension idim
				fc_idim = cell_centers[igrid * twotondimMndim + ind * ndim + idim] + sgn_face * dx_2;
				// Ray coordinate (z) of the intersection of the ray with the face plane
				z_intersect = (fc_idim - ray_orig_idim)/ ray_vect_idim;
				// Is this intersection within the face ?
				// Position of the face in the neighbour cell along the dimension idim (0 or 1)
				isgn = (-sgn_face+1)>>1;
				// Index of the son cell where the ray crosses the exit face (init.)
				ind_son = (twotondim-1) - (isgn<<idim);
				v=0;
				vl=0;
				for (idim2=0 ; idim2 < ndim ; idim2++){
					if (idim2 == idim) continue; // Skip the dimension perpendicular to the face
					// Coordinate of the intersection
					fcoord = (ray_origins[iray * ndim + idim2] + z_intersect * ray_vects[iray * ndim + idim2]);
					// Normalised coordinate within the face [ [-0.5, 0.5[, [-0.5, 0.5[ ]
					fcoordn = (fcoord - cell_centers[igrid * twotondimMndim + ind * ndim + idim2]) / dx_loc;
					if (fcoordn>vl) vl=fcoordn;
					if (fcoordn<0){
						// Index of the son cell where the ray crosses the exit face (update)
						ind_son -= (1<<idim2);
						fcoordn=-fcoordn;
					}
					if (fcoordn>v) v=fcoordn;
				}
				if ((v <= 0.500000001) && (vl<=0.500000001)){ // Intersection found
					test_ok=1;
					break;
				}
			}
		}
		if (test_ok)
		{
			//##################################################################
			if (z_intersect>=zmax){
				// End of ray-tracing
				z = zmax;
				zin = zmax;
			}
			else
				zin = z_intersect;
			// ##############   Ray integration loop  ##########################
			for (step=0 ; step < 1000000 ; step++){
				// Record current level
				ileveln = ilevel;
				indn = ind;
				//################// Find neighbour cell in octree ###################/
				if (((ind>>idim)&1) != isgn){
					iface = 2 * idim + 1 - isgn;
					igridn = neighbors[igrid * twoMdim + iface];
					if (cell_levels[igrid] != (ilevel+1) or igridn>ngrids){
						error = 1;
						break;
					}
					ileveln = cell_levels[igridn] - 1;
					// son index from ray/face intersection computation
					if (ileveln == ilevel){ // Same resolution neighbor cell
						// Neighbour cell index
						indn -= sgn_face * (1<<idim);
						if (ileveln+1 < max_read_level){
							// More refined neighbor cell ?
							igrid_son = sons[igridn * twotondim + indn];
							if (igrid_son > 0){
								igridn = igrid_son;
								ileveln += 1;
								// Neighbour cell index
								indn = ind_son - sgn_face * (1<<idim);
							}
						}
					}
					else if (ileveln == (ilevel-1)) {// Coarser resolution neighbor cell
						indn = 0;
						for (idim2=0 ; idim2 < ndim ; idim2++){
							cc = cell_centers[igrid * twotondimMndim + ind * ndim + idim2];
							if (idim2==idim) cc += sgn_face * dx_loc;
							if (cc>grid_centers[igridn * ndim + idim2]) indn += (1<<idim2);
						}
					}
					else { // Sanity check : neighbour cells should never be more than 1 level of
						// refinement apart (RAMSES octree property).
						error = 1;
						break;
					}
					igrid = igridn;
					ind = indn;
					ilevel = ileveln;
				}
				else {
					// Neighbour cell index in the current grid
					ind += sgn_face * (1<<idim); // neighbour cell index
					// More refined neighbor cell ?
					if (ilevel+1 < max_read_level){
						igrid_son = sons[igrid * twotondim + ind];
						if (igrid_son > 0){
							igrid = igrid_son;
							ilevel += 1;
							// Neighbour cell index
							ind = ind_son - sgn_face * (1<<idim);
						}
					}
				}
				// Fetch scalar data in neighbour cell
				scalar2 = scalar_data[igrid * twotondim + ind];
				scalarUp2 = scalar_data[twotondimMngrids + igrid * twotondim + ind];
				//########################################################//
	
				
				//################// Neighbour cell ray/face intersection computation ############
				dx_loc = dx[ilevel];
				dx_2 = dx[ilevel+1]; // dx_loc/2.0
				// Finding the exit intersection of the ray and the neighbour cell
	
				test_ok=0;
				for (idim=0 ; idim < ndim ; idim++){
					ray_vect_idim = ray_vects[iray * ndim + idim];
					ray_orig_idim = ray_origins[iray * ndim + idim];
					if(ray_vect_idim == 0) // ray is parallel to the faces perpendicular to dimension idim.
					{
						// There cannot be any intersection with this faces.
						continue;
					}
					else{
						if (ray_vect_idim > 0)
							sgn_face = 1;
						else
							sgn_face = -1;
		
						// Coordinate of the cell face center along dimension idim
						fc_idim = cell_centers[igrid * twotondimMndim + ind * ndim + idim] + sgn_face * dx_2;
						// Ray coordinate (z) of the intersection of the ray with the face plane
						z_intersect = (fc_idim - ray_orig_idim)/ ray_vect_idim;
						// Is this intersection within the face ?
						// Position of the face in the neighbour cell along the dimension idim (0 or 1)
						isgn = (-sgn_face+1)>>1;
						// Index of the son cell where the ray crosses the exit face (init.)
						ind_son = (twotondim-1) - (isgn<<idim);
						v=0;
						vl=0;
						for (idim2=0 ; idim2 < ndim ; idim2++){
							if (idim2 == idim) continue; // Skip the dimension perpendicular to the face
							// Coordinate of the intersection
							fcoord = (ray_origins[iray * ndim + idim2] + z_intersect * ray_vects[iray * ndim + idim2]);
							// Normalised coordinate within the face [ [-0.5, 0.5[, [-0.5, 0.5[ ]
							fcoordn = (fcoord - cell_centers[igrid * twotondimMndim + ind * ndim + idim2]) / dx_loc;
							if (fcoordn>vl) vl=fcoordn;
							if (fcoordn<0){
								// Index of the son cell where the ray crosses the exit face (update)
								ind_son -= (1<<idim2);
								fcoordn=-fcoordn;
							}
							if (fcoordn>v) v=fcoordn;
						}
						if ((v <= 0.500000001) && (vl<=0.500000001)){ // Intersection found
							test_ok=1;
							break;
						}
					}
				}
				if (not test_ok){
					//I[iray * 3 + 2] = 36; // to see error rays on lvlmax map
					break;//print "Error : no intersection found"
				}
				//##############################################################
	
				if (z_intersect>=zmax){
					// End of ray-tracing
					z = zmax;
				}
				//else if(z_intersect > zin) {
				else{
					zout = z_intersect;
					z = (zin +zout)/2.;
				}
				//else{ // this error case is commented to avoid additional openCL float precision error 
				//	error += 1;
				//	break;
				//}
				if (rgb){
					// Linear evolution of the scalar data
					ndz = (int) floor((z-zbefore)/(dx_min));
					if (ndz==0) ndz=1;
					if (ndz>5) ndz=5;
		//			ndz=1;
					dz = (z-zbefore) / ndz;
		//			dz_2 = dz / 2.0;
					zvar = zbefore;
					for (idz=0 ; idz < ndz ; idz++){
						scalar_loc = scalar;
						for (iem=0 ; iem < nlines ; iem++){
							x = scalar_loc - vlines[iem];
							width = wlines[iem];
							if (fabs(x) > (3.0*width))
							{
								if (x<0.0 ) break;
							}
							else{
		//						print gauss_value, scalar_loc
								gauss_value = exp(-x*x/(width*width));
								em = pow(10., (scalar_loc * gamma));
								rem = em * gauss_value * rlines[iem];
								gem = em * gauss_value * glines[iem];
								bem = em * gauss_value * blines[iem];
								alpha = gauss_value * alines[iem];
								absor = 1.0 - alpha * dz;
								if (absor>0.0){
									for (irgb=0 ; irgb < 3 ; irgb++)
										I[iray * 3 + irgb] = absor * I[iray * 3 + irgb];
								}
								I[iray * 3 + 0] += dz * rem;
								I[iray * 3 + 1] += dz * gem;
								I[iray * 3 + 2] += dz * bem;
								break;
							}
						}
						zvar += dz;
					}
				}
				else{
					// Piecewise constant evolution of the scalar field => dumb ray integration
					if (max_op){
						if (scalar > I[iray * 3 + 0])
							I[iray * 3 + 0] = scalar;
						if (scalarUp > I[iray * 3 + 1])
							I[iray * 3 + 1] = scalarUp;
					}
					else{
						I[iray * 3 + 0] += scalar * (zin-zbefore) + scalar2 * (z-zin);
						I[iray * 3 + 1] += scalarUp * (zin-zbefore) + scalarUp2 * (z-zin);
					}
					if (ilevel+1 > I[iray * 3 + 2])
						I[iray * 3 + 2] = ilevel+1;
				}
				zin = zout;
				scalar = scalar2;
				scalarUp = scalarUp2;
				zbefore = z;
				if (z>=zmax) break;
			}
			if (step == 1000000)
				error += 1;
			//#################### End of ray integration loop  ##########################
		}
	}
}
