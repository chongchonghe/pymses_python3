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
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "fio.h"
#include "utils.h"
#include "read_amr.h"


RAMSES_AmrHeader_t * RAMSES_AmrHeader_New()
{
	RAMSES_AmrHeader_t * hdr = malloc(sizeof(RAMSES_AmrHeader_t));
	hdr->out_times = NULL;
	hdr->out_aexps = NULL;
	return hdr;
}

void RAMSES_AmrHeader_Free(RAMSES_AmrHeader_t * header)
{
	FREE_IF_NOTNULL(header->out_times);
	FREE_IF_NOTNULL(header->out_aexps);
	free(header);
}

RAMSES_AmrStruct_t * RAMSES_AmrStruct_New()
{
	RAMSES_AmrStruct_t * amr = malloc(sizeof(RAMSES_AmrStruct_t));
	amr->ngridlevel = NULL;
	amr->ngridbound = NULL;
	amr->grid_indices = NULL;
	amr->coarse_son_indices = NULL;
	amr->grid_centers = NULL;
	amr->son_indices = NULL;
	return amr;
}


void RAMSES_AmrStruct_Free(RAMSES_AmrStruct_t * amr)
{
	FREE_IF_NOTNULL(amr->ngridlevel);
	FREE_IF_NOTNULL(amr->ngridbound);
	FREE_IF_NOTNULL(amr->grid_indices);
	FREE_IF_NOTNULL(amr->coarse_son_indices);
	FREE_IF_NOTNULL(amr->grid_centers);
	FREE_IF_NOTNULL(amr->son_indices);
	free(amr);
}

int RAMSES_AmrHeader_Read(RAMSES_AmrHeader_t * header, FIO_File * file)
{

	FIO_ReadRecord(&(header->ncpu), 4, 1, file);
	FIO_ReadRecord(&(header->ndim), 4, 1, file);
	FIO_ReadRecord(header->coarse_shape, 4, 3, file);

	int i;
	header->ncoarse = 1;
	for (i=0; i<3; i++)
		header->ncoarse *= header->coarse_shape[i];

	FIO_ReadRecord(&(header->levelmax), 4, 1, file);
	FIO_ReadRecord(&(header->ngridmax), 4, 1, file);
	FIO_ReadRecord(&(header->nboundary), 4, 1, file);

	FIO_ReadRecord(&(header->ngrids_lmax), 4, 1, file);
	FIO_ReadRecord(&(header->boxlen), 8, 1, file);

	int output_info[3];
	FIO_ReadRecord(output_info, 4, 3, file);
	header->noutputs = output_info[0];
	header->iout = output_info[1];

	header->out_times = malloc(sizeof(double)*header->noutputs);
	FIO_ReadRecord(header->out_times, 8, header->noutputs, file);

	header->out_aexps = malloc(sizeof(double)*header->noutputs);
	FIO_ReadRecord(header->out_aexps, 8, header->noutputs, file);
	FIO_ReadRecord(&(header->time), 8, 1, file);

	FIO_SkipRecord(file); /* dt_old */
	FIO_SkipRecord(file); /* dt_new */
	FIO_SkipRecord(file); /* nstep, nstep_coarse */
	FIO_SkipRecord(file); /* const, mass_tot_0, rho_tot */
	
	FIO_SkipRecord(file); /* cosmo parameters */

	double aexp_state[5];
	FIO_ReadRecord(aexp_state, 8, 5, file);
	header->aexp = aexp_state[0];

	FIO_SkipRecord(file); /* mass_sph */

	return 0;

}

int RAMSES_ReadAmr(
		RAMSES_AmrHeader_t * header, RAMSES_AmrStruct_t * amr,
		const char * amr_filename, int readlmax)
{


	/* Open the file */
	FIO_File * file;
	file = FIO_Open(amr_filename, "r", 4, FIO_DEFAULT);

	/* Read the header and fill some of the AMR structure */
	RAMSES_AmrHeader_Read(header, file);
	amr->ndim = header->ndim;
	amr->twotondim = 1<<amr->ndim;
	amr->ncpu = header->ncpu;
	amr->nboundary = header->nboundary;
	amr->levelmax = header->levelmax;
	amr->ncoarse = header->ncoarse;
	amr->readlmax = MIN(readlmax, amr->levelmax);

	int ndim = amr->ndim;
	int twotondim = amr->twotondim;
	int ncpu = amr->ncpu;
	int nboundary = amr->nboundary;
	int levelmax = amr->levelmax;

	/* Grid linked lists */
	FIO_SkipRecord(file); // headl
	FIO_SkipRecord(file); // taill

	/* Grids by level and CPU */
	int *tmp;
	tmp = malloc(sizeof(int)*ncpu*levelmax);
	FIO_ReadRecord(tmp, 4, ncpu*levelmax, file);

	/* Remap to C order */
	amr->ngridlevel = malloc(sizeof(int)*ncpu*levelmax);
	int i, j;
	for (i=0; i<ncpu; i++)
		for (j=0; j<levelmax; j++)
			CELEM2D(amr->ngridlevel,
				i, j, ncpu, levelmax) = FELEM2D(tmp, i, j, ncpu, levelmax);
	free(tmp);

	FIO_SkipRecord(file); /* numbtot */

	/* Boundary info */
	int xbound[3];

	if(nboundary > 0) {
		FIO_SkipRecord(file);  /* boundary linklist heads */
		FIO_SkipRecord(file);  /* boundary linklist tails */

		tmp = malloc(sizeof(int)*nboundary*levelmax);
		FIO_ReadRecord(tmp, 4, nboundary*levelmax, file);

		/* Remap to C order */
		amr->ngridbound = malloc(sizeof(int)*nboundary*levelmax);
		int i, j;
		for (i=0; i<nboundary; i++)
			for (j=0; j<levelmax; j++)
				CELEM2D(amr->ngridbound,
					i, j, nboundary, levelmax) = FELEM2D(tmp,
						i, j, nboundary, levelmax);
		free(tmp);
		for(i=0; i<3; i++)
			xbound[i] = header->coarse_shape[i]/2;
	} else {
		amr->ngridbound = NULL;
		for(i=0; i<3; i++)
			xbound[i] = 0;
	}

	FIO_SkipRecord(file); /* Free memory */
	FIO_SkipRecord(file); /* Ordering type */
	/* if ordering = bisection then
	 * FIO_SkipRecord(file);
	 * FIO_SkipRecord(file);
	 * FIO_SkipRecord(file);
	 * FIO_SkipRecord(file);
	 * else */
	FIO_SkipRecord(file); /* Hilbert bound keys */

	/* Cells at coarse level (level 0) */
	amr->coarse_son_indices = malloc(sizeof(int)*amr->ncoarse);
	FIO_ReadRecord(amr->coarse_son_indices, 4, amr->ncoarse, file);
	int icell;
	for(icell=0; icell<amr->ncoarse; icell++)
		amr->coarse_son_indices[icell] -= 1;

	FIO_SkipRecord(file); /* Coarse flags */
	FIO_SkipRecord(file); /* Coarse cpumap */

	/* Precompute total number of fine grids */
	int ilevel, icpu, cur_ngrids;
	int ngrids = 0;
	for (ilevel=0; ilevel<amr->readlmax; ilevel++) {
		for (icpu=0; icpu<ncpu+nboundary; icpu++)
			if(icpu<ncpu)
				ngrids += CELEM2D(amr->ngridlevel,
						icpu, ilevel, ncpu, levelmax);
			else
				ngrids += CELEM2D(amr->ngridbound,
						icpu-ncpu, ilevel, nboundary, levelmax);
	}
	amr->ngrids = ngrids;

	/* Temporary buffer for holding grid data records. Size is at most 8*ngrids,
	 * if data is written in double precision. */
	void * gbuf = malloc(8*ngrids);

	amr->grid_indices = malloc(sizeof(int)*ngrids);
	amr->grid_centers = malloc(sizeof(double)*ngrids*ndim);
	amr->son_indices = malloc(sizeof(int)*ngrids*twotondim);

	/* Loop over fine levels */
	int offset = 0;
	for (ilevel=0; ilevel<amr->readlmax; ilevel++) {

		for (icpu=0; icpu<ncpu+nboundary; icpu++) {

			if(icpu<ncpu)
				cur_ngrids = CELEM2D(amr->ngridlevel,
						icpu, ilevel, ncpu, levelmax);
			else
				cur_ngrids = CELEM2D(amr->ngridbound,
						icpu-ncpu, ilevel, nboundary, levelmax);

			if(cur_ngrids <= 0) continue;

			int idim, igrid;

			/* Grid index */
			FIO_ReadRecord(gbuf, 4, cur_ngrids, file);
			for (igrid=0; igrid<cur_ngrids; igrid++)
				amr->grid_indices[igrid+offset] = ((int*)gbuf)[igrid] - 1;

			/* Linked lists: prev and next */
			FIO_SkipRecord(file);
			FIO_SkipRecord(file);
			
			/* Grid centers */
			for(idim=0; idim<ndim; idim++) {
				FIO_ReadRecord(gbuf, 8, cur_ngrids, file);
				for (igrid=0; igrid<cur_ngrids; igrid++)
					CELEM2D(amr->grid_centers,
						igrid+offset, idim,
						ngrids, ndim) = ((double*)gbuf)[igrid] - xbound[idim];
			}

			/* Father index */
			FIO_SkipRecord(file);

			/* Neighbor index */
			int ind;
			for(ind=0; ind<2*ndim; ind++)
				FIO_SkipRecord(file);

			/* Son index */
			for(ind=0; ind<twotondim; ind++) {
				FIO_ReadRecord(gbuf, 4, cur_ngrids, file);
				for (igrid=0; igrid<cur_ngrids; igrid++) {
					int son = ((int*)gbuf)[igrid];
					/* Remap son index */
					son -= 1;
					CELEM2D(amr->son_indices,
						igrid+offset, ind,
						ngrids, twotondim) = son;
				}
			}
				
			/* Cell CPU map */
			for(ind=0; ind<twotondim; ind++)
				FIO_SkipRecord(file);

			/* Cell refinement map */
			for(ind=0; ind<twotondim; ind++)
				FIO_SkipRecord(file);

			offset += cur_ngrids;

		}
	}

	assert(offset == ngrids);

	/* Cleanup */
	free(gbuf);
	FIO_Close(file);

	return 0;
}
