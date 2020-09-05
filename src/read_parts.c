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
#include "fio.h"
#include "utils.h"
#include "read_parts.h"


RAMSES_PartData_t * RAMSES_PartData_New()
{
	RAMSES_PartData_t * part = malloc(sizeof(RAMSES_PartData_t));
	part->pos = NULL;
	part->vel = NULL;
	part->mass = NULL;
	part->id = NULL;
	part->level = NULL;
	part->epoch = NULL;
	return part;
}

void RAMSES_PartData_Free(RAMSES_PartData_t * part)
{
	FREE_IF_NOTNULL(part->pos);
	FREE_IF_NOTNULL(part->vel);
	FREE_IF_NOTNULL(part->mass);
	FREE_IF_NOTNULL(part->id);
	FREE_IF_NOTNULL(part->level);
	FREE_IF_NOTNULL(part->epoch);
	free(part);
}


int RAMSES_PartData_Read(RAMSES_PartData_t * part, double boxlen, FIO_File * file)
{

	/* Header */
	FIO_ReadRecord(&(part->ncpu), 4, 1, file);
	FIO_ReadRecord(&(part->ndim), 4, 1, file);
	FIO_ReadRecord(&(part->npart), 4, 1, file);
	FIO_SkipRecord(file); /* localseed */
	FIO_ReadRecord(&(part->nstar), 4, 1, file);
	FIO_SkipRecord(file); /* mstar */
	FIO_SkipRecord(file); /* mstar_lost */
	FIO_ReadRecord(&(part->nsink), 4, 1, file);

	int npart = part->npart;
	int ndim = part->ndim;

	int idim, ipart;
	double * dtmp = malloc(sizeof(double)*npart);

	/* Position */
	part->pos = malloc(sizeof(double)*npart*ndim);
	for(idim=0; idim<ndim; idim++) {
		FIO_ReadRecord(dtmp, 8, npart, file);
		for(ipart=0; ipart<npart; ipart++)
			CELEM2D(part->pos, ipart, idim, npart, ndim) = dtmp[ipart]/boxlen;
	}

	/* Velocity */
	part->vel = malloc(sizeof(double)*npart*ndim);
	for(idim=0; idim<ndim; idim++) {
		FIO_ReadRecord(dtmp, 8, npart, file);
		for(ipart=0; ipart<npart; ipart++)
			CELEM2D(part->vel, ipart, idim, npart, ndim) = dtmp[ipart];
	}

	/* Mass */
	part->mass = malloc(sizeof(double)*npart);
	FIO_ReadRecord(part->mass, 8, npart, file);

	/* Id */
	part->id = malloc(sizeof(int)*npart);
	FIO_ReadRecord(part->id, 4, npart, file);

	/* Level */
	part->level = malloc(sizeof(int)*npart);
	FIO_ReadRecord(part->level, 4, npart, file);

	/* Epoch */
	if(part->nstar>0 || part->nsink>0) {
		part->epoch = malloc(sizeof(double)*npart);
		FIO_ReadRecord(part->epoch, 8, npart, file);
	}

	/* Cleanup */
	free(dtmp);

	return 0;
}
