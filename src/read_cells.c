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
#include "read_cells.h"

RAMSES_CellData_t * RAMSES_CellData_New()
{
	RAMSES_CellData_t * cell;
	cell = malloc(sizeof(RAMSES_CellData_t));
	cell->data = NULL;
	cell->ivar_arr = NULL;
	return cell;
}


void RAMSES_CellData_Free(RAMSES_CellData_t * cell)
{
	if(cell->data != NULL) {
		int ivar;
		for(ivar=0; ivar<cell->nvar_data; ivar++)
			free(cell->data[ivar]);
		free(cell->data);
	}
	FREE_IF_NOTNULL(cell->ivar_arr);
	free(cell);
}


int RAMSES_CellData_Read(
		RAMSES_CellData_t * cell,
		const char * hydro_filename,
		RAMSES_CellData_FileType_t ftype,
		int readlmax,
		int ngrids,
		int nvarout,
		const int * ivarout_arr)
{

	/* Open the file */
	FIO_File * file;
	file = FIO_Open(hydro_filename, "r", 4, FIO_DEFAULT);

	/* Read the header */
	FIO_ReadRecord(&(cell->ncpu), 4, 1, file);

	if (ftype == RAMSES_CellData_HydroFile) {
		FIO_ReadRecord(&(cell->nvar_file), 4, 1, file);
		FIO_ReadRecord(&(cell->ndim), 4, 1, file);
	} else {
		FIO_ReadRecord(&(cell->ndim), 4, 1, file);
		cell->nvar_file = cell->ndim;
	}

	/* Copy the ivarout values into the structure */
	cell->nvar_data = nvarout;
	cell->ivar_arr = malloc(sizeof(int)*cell->nvar_data);
	int i;
	for(i=0; i<cell->nvar_data; i++)
		cell->ivar_arr[i] = ivarout_arr[i];

	FIO_ReadRecord(&(cell->levelmax), 4, 1, file);
	FIO_ReadRecord(&(cell->nboundary), 4, 1, file);

	if (ftype == RAMSES_CellData_HydroFile)
		FIO_SkipRecord(file); /* gamma */

	int twotondim = 1 << (cell->ndim);

	/* Allocate nvar_data cell-based 2D data arrays */
	cell->data = malloc(sizeof(double *)*cell->nvar_data);
	for(i=0; i<cell->nvar_data; i++)
		cell->data[i] = malloc(sizeof(double)*ngrids*twotondim);
	
	void * buf = malloc(sizeof(double)*ngrids);

	int ilevel;
	int offset = 0;

	for (ilevel=0; ilevel<readlmax; ilevel++) {

		int icpu;
		for (icpu=0; icpu<(cell->ncpu)+(cell->nboundary); icpu++) {

			FIO_SkipRecord(file); /* ilevel */

			int cur_ngrids;
			FIO_ReadRecord(&cur_ngrids, 4, 1, file);

			if(cur_ngrids <= 0) continue;

			int ind;
			for (ind=0; ind<twotondim; ind++) {
				int ivarfile;
				int ivarout = 0;
				for (ivarfile=0; ivarfile<cell->nvar_file; ivarfile++) {

					int more_to_read = (ivarout < nvarout);
					if (!more_to_read) {
						FIO_SkipRecord(file);
						continue;
					}

					int read_this = (ivarfile == ivarout_arr[ivarout]);
					if (!read_this) {
						FIO_SkipRecord(file);
						continue;
					}

					FIO_ReadRecord(buf, 8, cur_ngrids, file);
					int igrid;
					/* Dispatch the values in the proper data array */
					for (igrid=0; igrid<cur_ngrids; igrid++)
						CELEM2D(cell->data[ivarout],
								igrid+offset, ind,
								ngrids, twotondim) = ((double*)buf)[igrid];

					ivarout += 1;
				}
			}

			offset += cur_ngrids;

		}
	}

	free(buf);
	assert(offset == ngrids);

	/* Cleanup */
	FIO_Close(file);

	return 0;
}

