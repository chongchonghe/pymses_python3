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
#include"ray_trace_C_code.h"

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

static void ray_trace_C(double * maps, double * ray_length, double * ray_origins, double * ray_vects,
	double * cell_centers, int * sons, int * g_levels, double * scal_data,
	int * igrids, char * icells, long * active_mask, double * ray_depth_max,
	long * param_info_int){
	// This code is a C adapted copy of the original Cython code (in ray_trace.pyx)
	// See the Cython code for parameters comments
//	printf("ray_trace_C ! \n");
//	int j =0;
//	for (j =0;j<16;j++){
//		printf(" %i ",j);
//		printf("active_mask = %i \n",active_mask[j]);
//	}
	int nrays = param_info_int[0];
	int ndim = param_info_int[1];
	int nblocks = param_info_int[2];
	int max_read_level = param_info_int[3];
	int nfunc = param_info_int[4];
	int max_op = param_info_int[5];
	int use_dx = param_info_int[6];
	//int ngrids = param_info_int[7];
	
	int twotondim = 1 << ndim;
	int twotondimMndim = twotondim * ndim;
	int twotondimMnfunc = twotondim * nfunc;
	int igrid_root, level,iblock,idim,idim2,iray,head;
	int igrid, icell, igrid_son, ilevel, ifunc;
	int ind_son, intersect, sgn_face, block_already_done;
	int ind_son1, ind_son2, ind_entry, ind_exit, add, iblock2;
	char icell_root;
	double alpha, alpha_save, dx, dx_2, cube_depth, r2, l, d, v, vl;
	double Rsphere, Rsphere2;
	double sqrt3 = sqrt(3.0);
	double dm2, depth_max;

	int* igrid_pile = NULL;
	int* icell_pile = NULL;
	int* level_pile = NULL;
	int* dim_constr = NULL;
	int* dim_side = NULL;
	igrid_pile = malloc((twotondim-1) * max_read_level * sizeof(int));
	icell_pile = malloc((twotondim-1) * max_read_level * sizeof(int));
	level_pile = malloc((twotondim-1) * max_read_level * sizeof(int));
	dim_constr = malloc(ndim * sizeof(int));
	dim_side = malloc(ndim * sizeof(int));
	
	double* block_center = NULL;
	block_center = malloc(ndim * sizeof(double));
	
	for (iblock = 0 ; iblock < nblocks ; iblock++){
		igrid_root = igrids[iblock];
		icell_root = icells[iblock];
		block_already_done = 0;
		for (iblock2 = 0 ; iblock2 < iblock ; iblock2++){
			if ((igrids[iblock2] == igrids[iblock]) && (icells[iblock2] == icells[iblock])){
				block_already_done = 1;
				break;
			}
		}
		if (block_already_done == 1)
			continue;
		level = g_levels[igrid_root];
		dx = 1.0/(1<<level);
		Rsphere = sqrt3 * dx / 2.0;
		Rsphere2 = Rsphere*Rsphere;
		for (idim = 0 ; idim < ndim ; idim++){
			block_center[idim] = cell_centers[igrid_root * twotondimMndim + icell_root * ndim+ idim];
		}
//		#################
//		# Ray iteration #
//		#################
		for (iray = 0 ; iray < nrays ; iray++){
			depth_max = ray_depth_max[iray];
			dm2 = depth_max/2.0;
//			# Coordinate parameter of the point of the ray nearest to the block center
			alpha = 0.0;
			for (idim = 0 ; idim < ndim ; idim++){
				alpha += (block_center[idim]-ray_origins[iray * ndim + idim])*ray_vects[iray * ndim + idim];
			}
//			# square distances of the ray from the block center
			r2 = 0.0;
			for (idim = 0 ; idim < ndim ; idim++){
				v = ray_origins[iray * ndim + idim] + alpha * ray_vects[iray * ndim + idim] - block_center[idim];
				r2 += v*v;
			}
//			# Is this ray able to intersect the block ?
			if (r2 > Rsphere2)// # no => continue
				continue;
//			# distance from the ray extreme points
			d = alpha-dm2;
			if (d<0.)
				d=-d;
			d = d-dm2;
			if (d>0.) {
				r2 += d*d;
//				# Is this ray still able to intersect the block ?
				if (r2 > Rsphere2)
					continue;
			}
//			# Pile initialisation with current block
			head=0;
			igrid_pile[head] = igrid_root;
			icell_pile[head] = icell_root;
			level_pile[head] = level;
			while (head !=-1) {
				igrid = igrid_pile[head];
				icell = icell_pile[head];
				ilevel = level_pile[head];
				head -= 1;
/*				printf("head = %i ",head);*/
/*				printf("igrid = %i ",igrid);*/
/*				printf("icell = %i ",icell);*/
/*				printf("ilevel = %i \n",ilevel);*/
//				# Current cell size
				dx = 1.0/(1<<ilevel);
				dx_2 = dx/2;
//				# Cube optical depth computation
				cube_depth = 0.0;
				intersect = 0;
				for (idim2 = 0 ; idim2 < ndim ; idim2++){
					if(ray_vects[iray * ndim + idim2] == 0.0){
						// ray is parallel to the face along dimension idim.
						// There might not be an intersection.
						continue;
					}
					// All ray-cell intersections found => break loop
					if (intersect==2) break;
					for (sgn_face = -1 ; sgn_face < 2 ; sgn_face+=2){ // for sgn_face in [-1,1]:
//						# Coordinate parameter of the intersection of the ray with the face plane
						alpha = ((cell_centers[igrid * twotondimMndim + icell * ndim + idim2] + sgn_face * dx / 2.0)
							 - ray_origins[iray * ndim + idim2])/ray_vects[iray * ndim + idim2];
//							# Is this intersection within the face ?
						v=0.0;
						vl=0.0;
						ind_son = (twotondim-1) - (((-sgn_face+1)>>1)<<idim2);
						for (idim = 0 ; idim < ndim ; idim++){
							if (idim != idim2) {
								l = (ray_origins[iray * ndim + idim] + alpha * ray_vects[iray * ndim + idim]
								     - cell_centers[igrid * twotondimMndim + icell * ndim + idim])/dx;
								if ((l>vl) && (ray_vects[iray * ndim + idim] == 0.0))
									vl=l;
								if (l<0.0){
									l=-l;
									ind_son -= (1<<idim);
								}
								if (l>v)
									v=l;
							}
						}
						if ((v <= 0.5) && (vl < 0.5)){
//							# The intersection is in the face of the cube
							intersect +=1;
							if (intersect == 1){ //# First intersection found
								if (alpha < 0.0)
									alpha_save = 0.0;
								else if (alpha > depth_max)
									alpha_save = depth_max;
								else
									alpha_save = alpha;
								ind_son1=ind_son;
							}
							else{// # Second intersection found : done
								if (alpha < 0.0)
									alpha = 0.0;
								else if (alpha > depth_max)
									alpha = depth_max;
								cube_depth = alpha-alpha_save;
//									# Make sure ind_son1 = entry point son index
//									#           ind_son2 = exit point son index
								if (cube_depth < 0.0) {// # ind_son is the entry point son index
									cube_depth=-cube_depth;
									ind_son2 = ind_son1;
									ind_son1 = ind_son;
								}
								else//# ind_son is the exit point son index
									ind_son2=ind_son;
								break;
							}
						}
					}
				}
//				# Is the current cell actually intersected by the ray ?
				if (cube_depth > 0.0){
//					# The ray cuts the block
					igrid_son = sons[igrid * twotondim + icell];
					if ((igrid_son < 0) || (ilevel >= max_read_level)){
//					# We're done searching: the subcell is the one we want
//						# We skip inactive cells
						if (active_mask[igrid] == 0)
								continue;
						for (ifunc = 0 ; ifunc < nfunc ; ifunc++){
							if (use_dx)
								v = ilevel * 1.0;
							else
								v = scal_data[igrid * twotondimMnfunc + icell * nfunc + ifunc];
							ray_length[iray * nfunc + ifunc] += cube_depth;
							if (max_op){
								if (v>maps[iray * nfunc + ifunc])
									maps[iray * nfunc + ifunc] = v;
							}
							else
								maps[iray * nfunc + ifunc] += v * cube_depth;
						}
					}
					else{ // # Go down the AMR tree
//						#########################################################################################
//						#                      Add the NECESSARY son cells to the pile                          # 
//						#########################################################################################
						if (ind_son1 != ind_son2){
							for (idim = 0 ; idim < ndim ; idim++){
								ind_entry = (ind_son1>>idim) & 1;
								ind_exit = (ind_son2>>idim) & 1;
								if (ind_entry==ind_exit){
									dim_constr[idim] = 1;
									dim_side[idim] = ind_entry;
								}
								else
									dim_constr[idim] = 0;
							}

//							# Add the POSSIBLY intersecting son cells different from the entry/exit point son cells
							for (ind_son = 0 ; ind_son < twotondim ; ind_son++){ // # All he son cells
								add = 1;
//								# Different from entry/exit point son cells
								if ((ind_son == ind_son1)||(ind_son == ind_son2))
									add = 0;
								else{
									for (idim = 0 ; idim < ndim ; idim++){
										if ((dim_constr[idim])&&(dim_side[idim] != ((ind_son>>idim) & 1))){
											add = 0;
											break;
										}
									}
								}

								if (add){
									head += 1;
									level_pile[head] = ilevel+1;
									igrid_pile[head] = igrid_son;
									icell_pile[head] = ind_son;
								}
							}

//							# If different from the son cell of the entry point, add the exit point son cell
							head+=1;
							level_pile[head] = ilevel+1;
							igrid_pile[head] = igrid_son;
							icell_pile[head] = ind_son2;
						}

//						# Add the son cell of the entry point of the ray in the father grid
						head+=1;
						level_pile[head] = ilevel+1;
						igrid_pile[head] = igrid_son;
						icell_pile[head] = ind_son1;
					}
				}
			}
		}
	}
	free(igrid_pile);
	free(icell_pile);
	free(level_pile);
	free(dim_constr);
	free(dim_side);
	free(block_center);
//	printf("maps[0] = %f \n",maps[0]);
}

static void ray_trace_octree_C(double * I, double * ray_origins, double * ray_vects,
	double * grid_centers, double * cell_centers, int * sons, int * cell_levels,
	double * scalar_data, int * neighbors, double * ray_lengths,
	long * param_info_int, double gamma, double * vlines, double * wlines,
	double * rlines, double * glines, double * blines, double * alines){
	// This code is a C adapted copy of the original Cython code (in ray_trace.pyx)
	// See the Cython code for parameters comments
	int nrays = param_info_int[0];
	int ndim = param_info_int[1];
	int nlines = param_info_int[2];
	int max_read_level = param_info_int[3];
	int nfunc = param_info_int[4];
	int max_op = param_info_int[5];
	int ngrids = param_info_int[6];
	int rgb = param_info_int[7];
	int verbose = param_info_int[8];
	
	int twotondim = 1 << ndim;
	int twoMdim = 2 * ndim;
	int twotondimMndim = twotondim * ndim;
	int twotondimMngrids = twotondim * ngrids;
	
	// integer variables
	int iray, igrid, igridn, igrid_son, ilevel, ileveln, step;
	char idim, idim2, ind, indn, ibit, iface;
	char sgn_face, isgn, ind_son;

	// float variables
	double v, vl, t_0, t_1, tmin, tmax;
	double scalar, scalar2, cc, scalarUp, scalarUp2;
	double ray_vect_idim, ray_orig_idim, fc_idim, fcoord, fcoordn;
	double zbefore, z, zin, zout, z_intersect, zmax;
	
	double* dx = NULL;
	int level_max = 50;
	dx = malloc(level_max * sizeof(double));
	double dx_loc, dx_2;

	int test_ok;
	int idz, ndz, iem;
	double dx_min, dz, scalar_loc, zvar;
	double em, rem, gem, bem, alpha, gauss_value, absor, width, x;
	char irgb, error, ifunc;
	int errorInit, errorRay, errorTreeSearch;
	error = 0;
	errorInit = 0;
	errorRay = 0;
	errorTreeSearch = 0;
	//############// Precompute the cell sizes ################//
	dx[0]=0.5;
	for (ilevel=1 ; ilevel < level_max ; ilevel++){
		dx[ilevel] = dx[ilevel-1] / 2.0;
	}
	dx_min = dx[(max_read_level-1)];

	//#########################################################################/
	//                            Ray iteration                               //
	//#########################################################################/
	for (iray=0 ; iray < nrays ; iray++){ // Iteration over rays
		if (verbose && (iray%100000 == 0)){
			printf("ray = %i/", iray+1);
			printf("%i\n", nrays);
		}
		zmax = ray_lengths[iray];
		// Integration of the ray intensities (RGBA) between 0.0 (ray origin) and zmax
		
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
				errorInit += 1;
				continue;
			}
		}
		
		//#####/ Tree-search in octree to find where is the current ray origin #####/
		igrid = 0;
		igrid_son = 0;
		for (ilevel = 0 ; ilevel < max_read_level ; ilevel++){
				if (ilevel != 0) igrid = igrid_son;
				// Find son index in octree
				ind = 0;
				for (idim=0 ; idim < ndim ; idim++){
					ibit = (ray_origins[iray * ndim + idim] > grid_centers[igrid * ndim + idim]);
					ind += (ibit << idim);
				}
				igrid_son = sons[igrid * twotondim + ind];
				if (igrid_son == -1) break; // unsigned int - 1 = 4294967295
			}
		if (igrid_son >= 0 && ilevel == max_read_level){
			errorTreeSearch += 1;
			continue;
		}
		// for (ilevel = 0 ; ilevel <= 2 ; ilevel++){//max_read_level
			// ind = 0;
			// for (idim = 0 ; idim < ndim ; idim++){
				// ibit = ray_origins[iray * ndim + idim] > grid_centers[igrid*ndim + idim];
				// ind += (ibit << idim);
			// }
			// igrid_son = sons[igrid*twotondim + ind];
			// if (igrid_son < 0){ // unsigned int - 1 = 4294967295
				// break;
			// }
			// else{
				// igrid = igrid_son;
			// }
		// }
		// printf("ind %i ",ind);
		// printf("igrid_son %i\n ",igrid_son);
		// printf("ilevel %i\n ",ilevel);
		//####################################################
		
		// Now the ray_origin is in a leaf-cell of the current grid
		zbefore = 0.0;
		z=0.0; // Set the position along ray to 0.0 (ray origin)
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
			if ((ray_vect_idim == 0.0)) // ray is parallel to the faces perpendicular to dimension idim.
				// There cannot be any intersection with thes faces.
				continue;
			else if (ray_vect_idim > 0.0)
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
			v=0.0;
			vl=0.0;
			for (idim2=0 ; idim2 < ndim ; idim2++){
				if (idim2 == idim) continue; // Skip the dimension perpendicular to the face
				// Coordinate of the intersection
				fcoord = (ray_origins[iray * ndim + idim2] + z_intersect * ray_vects[iray * ndim + idim2]);
				// Normalised coordinate within the face [ [-0.5, 0.5[, [-0.5, 0.5[ ]
				fcoordn = (fcoord - cell_centers[igrid * twotondimMndim + ind * ndim + idim2]) / dx_loc;
				if (fcoordn>vl) vl=fcoordn;
				if (fcoordn<0.0){
					// Index of the son cell where the ray crosses the exit face (update)
					ind_son -= (1<<idim2);
					fcoordn=-fcoordn;
				}
				if (fcoordn>v) v=fcoordn;
			}
			if ((v <= 0.5) && (vl<=0.5)){ // Intersection found
				test_ok=1;
				break;
			}
		}
		// printf("ilevel %i\n ",ilevel);
//		if (not test_ok)
//			print "Error : no intersection found"
//			return
		//##################################################################

//		I[iray * 3 + 0] = ilevel+1;
		
		if (z_intersect>=zmax){
			// End of ray-tracing
			z = zmax;
			zin = zmax;
		}
		else
			zin = z_intersect;
		step = 0;
		while (z<zmax && step < 1000000){ // ##############   Ray integration loop  ##########################
			step += 1;
			// Record current level
			ileveln = ilevel;
			indn = ind;
			//################// Find neighbour cell in octree ###################/
			if (((ind>>idim)&1) != isgn){
				iface = 2 * idim + 1 - isgn;
				igridn = neighbors[igrid * twoMdim + iface];
				if ((cell_levels[igrid] != (ilevel+1)) || (igridn>ngrids) || (igridn<0)){
					error = 1;
					//printf("%i, ", cell_levels[igrid]);
					//printf("%i\n", (ilevel+1));
					//I[iray * 3 + 2] = 36; // to see error rays on lvlmax map
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
					//print "level error : (ilevel, igridn, ilevel_neighbour) = ", ilevel, igridn, ileveln
					//print "              (zin, z, zout) = ", z, zin, zout
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
				if(ray_vect_idim == 0.0) // ray is parallel to the faces perpendicular to dimension idim.
				{
					// There cannot be any intersection with this faces.
					continue;
				}
				else if (ray_vect_idim > 0.0)
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
				v=0.0;
				vl=0.0;
				for (idim2=0 ; idim2 < ndim ; idim2++){
					if (idim2 == idim) continue; // Skip the dimension perpendicular to the face
					// Coordinate of the intersection
					fcoord = (ray_origins[iray * ndim + idim2] + z_intersect * ray_vects[iray * ndim + idim2]);
					// Normalised coordinate within the face [ [-0.5, 0.5[, [-0.5, 0.5[ ]
					fcoordn = (fcoord - cell_centers[igrid * twotondimMndim + ind * ndim + idim2]) / dx_loc;
					if (fcoordn>vl) vl=fcoordn;
					if (fcoordn<0.0){
						// Index of the son cell where the ray crosses the exit face (update)
						ind_son -= (1<<idim2);
						fcoordn=-fcoordn;
					}
					if (fcoordn>v) v=fcoordn;
				}
				if ((v <= 0.5) && (vl<=0.5)){ // Intersection found
					test_ok=1;
					break;
				}
			}
			if (!test_ok){
				error = 1;
				//I[iray * 3 + 2] = 36; // to see error rays on lvlmax map
				//print "Error : no intersection found"
			}
			//##############################################################

			if (z_intersect>=zmax){
				// End of ray-tracing
				z = zmax;
			}
			else if(z_intersect > zin) {
				zout = z_intersect;
				z = (zin +zout)/2.;
			}
			else{
				errorRay += 1;
				break;
			}
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
		}
		if (step == 1000000)
			errorRay += 1;
		//#################### End of ray integration loop  ##########################
	}
	//############################ End of ray iteration ##############################
	if (error)
		printf("OCTREE ERRORS OCCURED DURING RAY TRACING...\n");
	if (errorRay)
		printf("errorRay = %i...\n", errorRay);
	//if (errorInit)
	//	printf("errorInit = %i...\n", errorInit);
	if (errorTreeSearch)
		printf("errorTreeSearch = %i...\n", errorTreeSearch);

	free(dx);
}