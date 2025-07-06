/*
 * Gadget-2 HDF5 I/O for Rockstar
 * Minh Nguyen (nhat.minh.nguyen@ipmu.jp)
 */
#ifdef ENABLE_HDF5

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <hdf5.h> /* HDF5 required */
#include "io_hdf5.h"
#include "io_gadget2_hdf5.h"
#include "io_util.h"
#include "../universal_constants.h"
#include "../check_syscalls.h"
#include "../config_vars.h"
#include "../config.h"
#include "../particle.h"

void gadget2_readdataset_float(hid_t HDF_FileID, char *filename, char *gid,
                               char *dataid, struct particle *p, int64_t to_read,
                               int64_t offset, int64_t stride) {
    int64_t   width   = H5Tget_size(H5T_NATIVE_FLOAT);
    void     *buffer  = check_malloc_s(buffer, to_read, width * stride);
    float    *fbuffer = buffer;

    hid_t HDF_GroupID     = check_H5Gopen(HDF_FileID, gid, filename);
    hid_t HDF_DatasetID   = check_H5Dopen(HDF_GroupID, dataid, gid, filename);
    hid_t HDF_DataspaceID = check_H5Dget_space(HDF_DatasetID);

    check_H5Sselect_all(HDF_DataspaceID);
    hssize_t npoints = H5Sget_select_npoints(HDF_DataspaceID);

    if (npoints != to_read * stride) {
        fprintf(stderr,
                "[Error] dataspace %s/%s in HDF5 file %s not expected size!\n  "
                "(Actual size = %" PRId64 " elements; expected size = %" PRId64
                " elements\n",
                gid, dataid, filename, (int64_t)(npoints), stride * to_read);
        exit(1);
    }

    check_H5Dread(HDF_DatasetID, H5T_NATIVE_FLOAT, buffer, dataid, gid, filename);

    H5Sclose(HDF_DataspaceID);
    H5Dclose(HDF_DatasetID);
    H5Gclose(HDF_GroupID);

    for (int64_t i = 0; i < to_read; i++)
        memcpy(((char *)&(p[i])) + offset, fbuffer + (i * stride), stride * width);

    free(buffer);
}

void gadget2_readdataset_ID_uint32(hid_t HDF_FileID, char *filename, char *gid,
                                   char *dataid, struct particle *p, int64_t to_read,
                                   int64_t offset, int64_t stride) {
    int64_t   width   = H5Tget_size(H5T_NATIVE_UINT32);
    void     *buffer  = check_malloc_s(buffer, to_read, width * stride);
    uint32_t *ibuffer = buffer;

    hid_t HDF_GroupID     = check_H5Gopen(HDF_FileID, gid, filename);
    hid_t HDF_DatasetID   = check_H5Dopen(HDF_GroupID, dataid, gid, filename);
    hid_t HDF_DataspaceID = check_H5Dget_space(HDF_DatasetID);

    check_H5Sselect_all(HDF_DataspaceID);
    hssize_t npoints = H5Sget_select_npoints(HDF_DataspaceID);

    if (npoints != to_read * stride) {
        fprintf(stderr,
                "[Error] dataspace %s/%s in HDF5 file %s not expected size!\n  "
                "(Actual size = %" PRId64 " elements; expected size = %" PRId64
                " elements\n",
                gid, dataid, filename, (int64_t)(npoints), stride * to_read);
        exit(1);
    }

    check_H5Dread(HDF_DatasetID, H5T_NATIVE_UINT32, buffer, dataid, gid, filename);

    H5Sclose(HDF_DataspaceID);
    H5Dclose(HDF_DatasetID);
    H5Gclose(HDF_GroupID);

    for (int64_t i = 0; i < to_read; i++)
        p[i].id = (int64_t) ibuffer[i];

    free(buffer);
}

void gadget2_readdataset_ID_uint64(hid_t HDF_FileID, char *filename, char *gid,
                                   char *dataid, struct particle *p, int64_t to_read,
                                   int64_t offset, int64_t stride) {
    int64_t   width   = H5Tget_size(H5T_NATIVE_UINT64);
    void     *buffer  = check_malloc_s(buffer, to_read, width * stride);
    uint64_t *ibuffer = buffer;

    hid_t HDF_GroupID     = check_H5Gopen(HDF_FileID, gid, filename);
    hid_t HDF_DatasetID   = check_H5Dopen(HDF_GroupID, dataid, gid, filename);
    hid_t HDF_DataspaceID = check_H5Dget_space(HDF_DatasetID);

    check_H5Sselect_all(HDF_DataspaceID);
    hssize_t npoints = H5Sget_select_npoints(HDF_DataspaceID);

    if (npoints != to_read * stride) {
        fprintf(stderr,
                "[Error] dataspace %s/%s in HDF5 file %s not expected size!\n  "
                "(Actual size = %" PRId64 " elements; expected size = %" PRId64
                " elements\n",
                gid, dataid, filename, (int64_t)(npoints), stride * to_read);
        exit(1);
    }

    check_H5Dread(HDF_DatasetID, H5T_NATIVE_UINT64, buffer, dataid, gid, filename);

    H5Sclose(HDF_DataspaceID);
    H5Dclose(HDF_DatasetID);
    H5Gclose(HDF_GroupID);

    for (int64_t i = 0; i < to_read; i++)
        p[i].id = (int64_t) ibuffer[i];

    free(buffer);
}

double gadget2_readheader_double(hid_t HDF_GroupID, char *filename, char *objName) {
    char *gid        = "Header";
    hid_t HDF_Type   = H5T_NATIVE_DOUBLE;
    hid_t HDF_AttrID = check_H5Aopen_name(HDF_GroupID, objName, gid, filename);
    hid_t HDF_DataspaceID = check_H5Aget_space(HDF_AttrID);

    check_H5Sselect_all(HDF_DataspaceID);

    double data = 0.0;
    check_H5Aread(HDF_AttrID, HDF_Type, &data, objName, gid, filename);

    H5Sclose(HDF_DataspaceID);
    H5Aclose(HDF_AttrID);
    return data;
}

void gadget2_readheader_array(hid_t HDF_GroupID, char *filename, char *objName,
                              hid_t type, void *data) {
    char *gid        = "Header";
    hid_t HDF_AttrID = check_H5Aopen_name(HDF_GroupID, objName, gid, filename);
    hid_t HDF_DataspaceID = check_H5Aget_space(HDF_AttrID);
    check_H5Sselect_all(HDF_DataspaceID);

    int64_t ndims = check_H5Sget_simple_extent_ndims(HDF_DataspaceID);
    assert(ndims == 1);
    hsize_t dimsize = 0;
    check_H5Sget_simple_extent_dims(HDF_DataspaceID, &dimsize);
    assert(dimsize == GADGET2_NTYPES);

    check_H5Aread(HDF_AttrID, type, data, objName, gid, filename);

    H5Sclose(HDF_DataspaceID);
    H5Aclose(HDF_AttrID);
}

void gadget2_hdf5_rescale_particles(struct particle *p, int64_t p_start,
                               int64_t nelems) {
    double vel_rescale = sqrt(SCALE_NOW);
    if (LIGHTCONE)
        vel_rescale = 1;

    for (int64_t i = 0; i < nelems; i++) {
        for (int64_t j = 0; j < 3; j++) {
            p[p_start + i].pos[j] *= GADGET2_LENGTH_CONVERSION;
            p[p_start + i].pos[j + 3] *= vel_rescale;
        }
    }
}

void load_particles_gadget2_hdf5(char *filename, struct particle **p, int64_t *num_p) {
    hid_t HDF_FileID = check_H5Fopen(filename, H5F_ACC_RDONLY);
    hid_t HDF_Header = check_H5Gopen(HDF_FileID, "Header", filename);

    Ol        = gadget2_readheader_double(HDF_Header, filename, "OmegaLambda");
    Om        = gadget2_readheader_double(HDF_Header, filename, "Omega0");
    h0        = gadget2_readheader_double(HDF_Header, filename, "HubbleParam");
    SCALE_NOW = gadget2_readheader_double(HDF_Header, filename, "Time");
    BOX_SIZE  = gadget2_readheader_double(HDF_Header, filename, "BoxSize");
    BOX_SIZE *= GADGET2_LENGTH_CONVERSION;

    uint64_t npart[GADGET2_NTYPES], npart_total[GADGET2_NTYPES], npart_total_hi[GADGET2_NTYPES];
    float    massTable[GADGET2_NTYPES];

    gadget2_readheader_array(HDF_Header, filename, "NumPart_ThisFile",
                             H5T_NATIVE_UINT64, npart);
    gadget2_readheader_array(HDF_Header, filename, "NumPart_Total",
                             H5T_NATIVE_UINT64, npart_total);
	gadget2_readheader_array(HDF_Header, filename, "NumPart_Total_HighWord",
                         	 H5T_NATIVE_UINT64, npart_total_hi);
    gadget2_readheader_array(HDF_Header, filename, "MassTable", H5T_NATIVE_FLOAT,
                             massTable);

    uint64_t TOTAL_PARTICLES = npart_total[GADGET2_DM_PARTTYPE]+(npart_total_hi[GADGET2_DM_PARTTYPE] << 32);

    H5Gclose(HDF_Header);
    H5Gclose(HDF_Header);

	PARTICLE_MASS = massTable[GADGET2_DM_PARTTYPE] * GADGET2_MASS_CONVERSION;

	if (RESCALE_PARTICLE_MASS) {
		PARTICLE_MASS = Om * CRITICAL_DENSITY * pow(BOX_SIZE, 3) / TOTAL_PARTICLES;
	}

    AVG_PARTICLE_SPACING = cbrt(PARTICLE_MASS / (Om * CRITICAL_DENSITY));

    printf("GADGET2: filename:       %s\n", filename);
    printf("GADGET2: box size:       %g Mpc/h\n", BOX_SIZE);
    printf("GADGET2: h0:             %g\n", h0);
    printf("GADGET2: scale factor:   %g\n", SCALE_NOW);
    printf("GADGET2: Total DM Part:  %" PRIu64 "\n", TOTAL_PARTICLES);
    printf("GADGET2: ThisFile DM Part: %" PRIu64 "\n", npart[GADGET2_DM_PARTTYPE]);
    printf("GADGET2: DM Part Mass:   %g Msun/h\n", PARTICLE_MASS);
    printf("GADGET2: avgPartSpacing: %g Mpc/h\n\n", AVG_PARTICLE_SPACING);

    if (!npart[GADGET2_DM_PARTTYPE]) {
        H5Fclose(HDF_FileID);
        printf("   SKIPPING FILE, PARTICLE COUNT ZERO.\n");
        return;
    }

    int64_t to_read = npart[GADGET2_DM_PARTTYPE];
    check_realloc_s(*p, ((*num_p) + to_read), sizeof(struct particle));

    // read IDs, pos, vel
    char buffer[100];
    snprintf(buffer, 100, "PartType%" PRId64, GADGET2_DM_PARTTYPE);

    if (GADGET2_ID_BYTES == 8) {
        gadget2_readdataset_ID_uint64(
            HDF_FileID, filename, buffer, "ParticleIDs", *p + (*num_p), to_read,
            (char *)&(p[0][0].id) - (char *)(p[0]), 1);
	}
    else if (GADGET2_ID_BYTES == 4) {
        gadget2_readdataset_ID_uint32(
            HDF_FileID, filename, buffer, "ParticleIDs", *p + (*num_p), to_read,
            (char *)&(p[0][0].id) - (char *)(p[0]), 1);
    }
    else {
        fprintf(stderr, "[Error] Unrecognized GADGET2_ID_BYTES:%d\n", (int) GADGET2_ID_BYTES);
        exit(1);
    }

    gadget2_readdataset_float(
        HDF_FileID, filename, buffer, "Coordinates", *p + (*num_p), to_read,
        (char *)&(p[0][0].pos[0]) - (char *)(p[0]), 3);
    gadget2_readdataset_float(
        HDF_FileID, filename, buffer, "Velocities", *p + (*num_p), to_read,
        (char *)&(p[0][0].pos[3]) - (char *)(p[0]), 3);

    H5Fclose(HDF_FileID);

    gadget2_hdf5_rescale_particles(*p, *num_p, to_read);

    *num_p += npart[GADGET2_DM_PARTTYPE];
}

#endif /* ENABLE_HDF5 */
