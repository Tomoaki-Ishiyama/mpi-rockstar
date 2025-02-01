#ifdef ENABLE_HDF5
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <hdf5.h> /* HDF5 required */
#include "meta_io.h"
#include "io_internal.h"
#include "io_hdf5.h"
#include "../config_vars.h"
#include "../rockstar.h"
#include "../groupies.h"
#include "../check_syscalls.h"
#include "../version.h"
#include "../halo.h"

void set_buffer(void *buffer, const int *to_write,
                int64_t offset, int64_t stride, hsize_t type) {
    int64_t *ibuffer = buffer;
    float   *fbuffer = buffer;
    int64_t i, id, width;

    if (type != H5T_NATIVE_FLOAT && type != H5T_NATIVE_LLONG) {
        fprintf(stderr, "[Error] set_buffer accepts only H5T_NATIVE_FLOAT and H5T_NATIVE_LLONG!\n");
        exit(1);
    }

    if (type == H5T_NATIVE_FLOAT) width = 4;
    if (type == H5T_NATIVE_LLONG) width = 8;

    for (i = 0, id = 0; i < num_halos; i++) {
        if (!to_write[i])
            continue;
        if (type == H5T_NATIVE_FLOAT)
            memcpy(fbuffer + (id * stride), ((char *) &(halos[i])) + offset, stride * width);
        if (type == H5T_NATIVE_LLONG)
            memcpy(ibuffer + (id * stride), ((char *) &(halos[i])) + offset, stride * width);
        id++;
    }
}

void load_buffer(void *buffer, struct halo *halos, int64_t to_read,
                 int64_t offset, int64_t stride, hsize_t type) {
    int64_t *ibuffer = buffer;
    float   *fbuffer = buffer;
    int64_t i, width;

    if (type != H5T_NATIVE_FLOAT && type != H5T_NATIVE_LLONG) {
        fprintf(stderr, "[Error] load_buffer accepts only H5T_NATIVE_FLOAT and H5T_NATIVE_LLONG!\n");
        exit(1);
    }

    if (type == H5T_NATIVE_FLOAT) width = 4;
    if (type == H5T_NATIVE_LLONG) width = 8;

    for (i = 0; i < to_read; i++) {
        if (type == H5T_NATIVE_FLOAT)
            memcpy(((char *) &(halos[i])) + offset, fbuffer + (i * stride), stride * width);
        if (type == H5T_NATIVE_LLONG)
            memcpy(((char *) &(halos[i])) + offset, ibuffer + (i * stride), stride * width);
    }
}

void write_hdf5_dataset(hid_t hid, char *dataid, hid_t type,
                        hsize_t rank, hsize_t *dims, void *data) {
    hid_t HDF_DataspaceID = check_H5Screate_simple(rank, dims, NULL); 
    hid_t HDF_DatasetID   = check_H5Dcreate(hid, dataid, type, HDF_DataspaceID);

    check_H5Dwrite(HDF_DatasetID, type, data);
    check_H5Dclose(HDF_DatasetID);
    check_H5Sclose(HDF_DataspaceID);
}

void add_hdf5_attribute(hid_t hid, char *dataid,
                        char *unit, char *description) {
    hid_t dataspace_id, attrtype_id, attrspace_id, attr_id;

    dataspace_id = check_H5Dopen2(hid, dataid);

    attrtype_id = check_H5Tcopy(H5T_C_S1);
    check_H5Tset_size(attrtype_id, 256);
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(dataspace_id, "Unit", attrtype_id, attrspace_id);
    check_H5Awrite(attr_id, attrtype_id, unit);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);
    check_H5Tclose(attrtype_id);

    attrtype_id = check_H5Tcopy(H5T_C_S1);
    check_H5Tset_size(attrtype_id, 256);
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(dataspace_id, "Description", attrtype_id, attrspace_id);
    check_H5Awrite(attr_id, attrtype_id, description);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);
    check_H5Tclose(attrtype_id);

    check_H5Dclose(dataspace_id);
}

void read_hdf5_dataset(hid_t hid, char *dataid,
                       hid_t type, void *buffer) {
    hid_t HDF_DatasetID   = check_H5Dopen2(hid, dataid);
    hid_t HDF_DataspaceID = check_H5Dget_space(HDF_DatasetID);

    check_H5Sselect_all(HDF_DataspaceID);
    check_H5Dread2(HDF_DatasetID, type, buffer, dataid);
    H5Sclose(HDF_DataspaceID);
    H5Dclose(HDF_DatasetID);
}

void read_hdf5_halos(hid_t HDF_FileID, struct halo *halos,
                     struct binary_output_header *bh) {
    int64_t *buffer_int;
    float   *buffer_float;
    int64_t  num_halos;
    char     dataid[256];
    hid_t    HDF_GroupID;

    num_halos = bh->num_halos;
    buffer_float = (float *) malloc(sizeof(float) * 6 * num_halos);
    buffer_int = (int64_t *) malloc(sizeof(int64_t) * num_halos);

    HDF_GroupID = check_H5Gopen2(HDF_FileID, "/Subhalo");

    read_hdf5_dataset(HDF_GroupID, "ID", H5T_NATIVE_LLONG, buffer_int);
    load_buffer(buffer_int, halos, num_halos,
                (char *) &(halos[0].id) - (char *) (halos), 1, H5T_NATIVE_LLONG);

    read_hdf5_dataset(HDF_GroupID, "ParticleStart", H5T_NATIVE_LLONG, buffer_int);
    load_buffer(buffer_int, halos, num_halos,
                (char *) &(halos[0].p_start) - (char *) (halos), 1, H5T_NATIVE_LLONG);

    read_hdf5_dataset(HDF_GroupID, "Position", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].pos[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "Velocity", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].pos[3]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "CoreVelocity", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].corevel[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "BulkVelocity", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].bulkvel[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);

    sprintf(dataid, "M%s", MASS_DEFINITION);
    read_hdf5_dataset(HDF_GroupID, dataid, H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].m) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    sprintf(dataid, "R%s", MASS_DEFINITION);
    read_hdf5_dataset(HDF_GroupID, dataid, H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].r) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "ChildRadius", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].child_r) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "RadiusDynamicalVirial", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].vmax_r) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "Mbound", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].mgrav) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "Vmax", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].vmax) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

#ifdef OUTPUT_RVMAX
    read_hdf5_dataset(HDF_GroupID, "VmaxRadius", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].rvmax) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
#endif

#ifdef OUTPUT_NFW_CHI2
    read_hdf5_dataset(HDF_GroupID, "NFWChi2", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].chi2) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
#endif

    read_hdf5_dataset(HDF_GroupID, "ScaleRadius", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].rs) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "ScaleRadiusKlypin", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].klypin_rs) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "Vrms", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].vrms) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "AngularMomentum", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].J[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "Energy", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].energy) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "Spin", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].spin) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    sprintf(dataid, "M%s", MASS_DEFINITION2);
    read_hdf5_dataset(HDF_GroupID, dataid, H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].alt_m[0]) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    sprintf(dataid, "M%s", MASS_DEFINITION3);
    read_hdf5_dataset(HDF_GroupID, dataid, H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].alt_m[1]) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    sprintf(dataid, "M%s", MASS_DEFINITION4);
    read_hdf5_dataset(HDF_GroupID, dataid, H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].alt_m[2]) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    sprintf(dataid, "M%s", MASS_DEFINITION5);
    read_hdf5_dataset(HDF_GroupID, dataid, H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].alt_m[3]) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "PositionOffset", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].Xoff) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "VelocityOffset", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].Voff) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "SpinBullock", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].bullock_spin) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "KineticToPotentialRatio", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].kin_to_pot) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "MpeBehroozi", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].m_pe_b) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "MpeDiemer", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].m_pe_d) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "HalfMassRadius", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].halfmass_radius) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "NumberParticles", H5T_NATIVE_LLONG, buffer_int);
    load_buffer(buffer_int, halos, num_halos,
                (char *) &(halos[0].num_p) - (char *) (halos), 1, H5T_NATIVE_LLONG);

    read_hdf5_dataset(HDF_GroupID, "NumberChildParticles", H5T_NATIVE_LLONG, buffer_int);
    load_buffer(buffer_int, halos, num_halos,
                (char *) &(halos[0].num_child_particles) - (char *) (halos), 1, H5T_NATIVE_LLONG);

    read_hdf5_dataset(HDF_GroupID, "DescendantID", H5T_NATIVE_LLONG, buffer_int);
    load_buffer(buffer_int, halos, num_halos,
                (char *) &(halos[0].desc) - (char *) (halos), 1, H5T_NATIVE_LLONG);

    read_hdf5_dataset(HDF_GroupID, "Flags", H5T_NATIVE_LLONG, buffer_int);
    load_buffer(buffer_int, halos, num_halos,
                (char *) &(halos[0].flags) - (char *) (halos), 1, H5T_NATIVE_LLONG);

    read_hdf5_dataset(HDF_GroupID, "NumberCoreParticles", H5T_NATIVE_LLONG, buffer_int);
    load_buffer(buffer_int, halos, num_halos,
                (char *) &(halos[0].n_core) - (char *) (halos), 1, H5T_NATIVE_LLONG);

    check_H5Gclose(HDF_GroupID);

    HDF_GroupID = check_H5Gopen2(HDF_FileID, "/Shape");

    read_hdf5_dataset(HDF_GroupID, "IntermediateToMajorAxisRatio", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].b_to_a) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "MinorToMajorAxisRatio", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].c_to_a) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "IntermediateToMajorAxisRatio2", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].b_to_a2) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "MinorToMajorAxisRatio2", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].c_to_a2) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "MajorAxis", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].A[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "MajorAxis2", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].A2[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);

#ifdef OUTPUT_INTERMEDIATE_AXIS
    read_hdf5_dataset(HDF_GroupID, "IntermediateAxis", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].A_I[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "IntermediateAxis2", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].A2_I[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);
#endif

#ifdef OUTPUT_INERTIA_TENSOR
    read_hdf5_dataset(HDF_GroupID, "InertiaTensor", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].inertia_tensor[0]) - (char *) (halos), 6, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "InertiaTensor2", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].inertia_tensor2[0]) - (char *) (halos), 6, H5T_NATIVE_FLOAT);
#endif

    check_H5Gclose(HDF_GroupID);

    HDF_GroupID = check_H5Gopen2(HDF_FileID, "/Error");

    read_hdf5_dataset(HDF_GroupID, "MinPosError", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].min_pos_err) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "MinVelError", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].min_vel_err) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    read_hdf5_dataset(HDF_GroupID, "MinBulkVelError", H5T_NATIVE_FLOAT, buffer_float);
    load_buffer(buffer_float, halos, num_halos,
                (char *) &(halos[0].min_bulkvel_err) - (char *) (halos), 1, H5T_NATIVE_FLOAT);

    check_H5Gclose(HDF_GroupID);

    free(buffer_float);
    free(buffer_int);
}

void write_hdf5_header(hid_t HDF_FileID, struct binary_output_header *bh,
                       int64_t tot_num_halos, int64_t tot_num_p) {
    hid_t attr_id, attrspace_id, attrtype_id;
    hsize_t dim[1];
    int64_t ibuffer;
    float buffer;

    hid_t group_id = check_H5Gcreate(HDF_FileID, "/Header");

    // Magic
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "Magic", H5T_NATIVE_ULLONG, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_ULLONG, &bh->magic);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // Snap
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "Snap", H5T_NATIVE_LLONG, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_LLONG, &bh->snap);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // Chunk
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "Chunk", H5T_NATIVE_LLONG, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_LLONG, &bh->chunk);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // ScaleFactor
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "ScaleFactor", H5T_NATIVE_FLOAT, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_FLOAT, &bh->scale);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // Omega_m
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "Omega_m", H5T_NATIVE_FLOAT, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_FLOAT, &bh->Om);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // Omega_L
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "Omega_L", H5T_NATIVE_FLOAT, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_FLOAT, &bh->Ol);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // H0
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "h", H5T_NATIVE_FLOAT, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_FLOAT, &bh->h0);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // Bounds
    attrspace_id = check_H5Screate(H5S_SIMPLE);
    dim[0] = 6;
    check_H5Sset_extent_simple(attrspace_id, 1, dim, NULL);
    attr_id = check_H5Acreate(group_id, "Bounds", H5T_NATIVE_FLOAT, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_FLOAT, &bh->bounds[0]);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // NumberHalos
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "NumberHalos", H5T_NATIVE_LLONG, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_LLONG, &bh->num_halos);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // NumberParticles
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "NumberParticles", H5T_NATIVE_LLONG, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_LLONG, &bh->num_particles);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // BoxSize
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "BoxSize", H5T_NATIVE_FLOAT, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_FLOAT, &bh->box_size);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // ParticleMass
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "ParticleMass", H5T_NATIVE_FLOAT, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_FLOAT, &bh->particle_mass);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // ParticleType
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "ParticleType", H5T_NATIVE_LLONG, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_LLONG, &bh->particle_type);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // FormatRevision
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "FormatRevision", H5T_NATIVE_INT, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_INT, &bh->format_revision);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // Version
    attrtype_id = check_H5Tcopy(H5T_C_S1);
    check_H5Tset_size(attrtype_id, VERSION_MAX_SIZE);
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "Version", attrtype_id, attrspace_id);
    check_H5Awrite(attr_id, attrtype_id, &bh->rockstar_version[0]);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);
    check_H5Tclose(attrtype_id);

    /* Items below are not stored in binary_header. */
    // ForceResolution
    attrspace_id = check_H5Screate(H5S_SCALAR);
    buffer = FORCE_RES;
    attr_id = check_H5Acreate(group_id, "ForceResolution", H5T_NATIVE_FLOAT, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_FLOAT, &buffer);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // FOFLinkingLength
    attrspace_id = check_H5Screate(H5S_SCALAR);
    buffer = FOF_LINKING_LENGTH;
    attr_id = check_H5Acreate(group_id, "FOFLinkingLength", H5T_NATIVE_FLOAT, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_FLOAT, &buffer);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // UnboundThreshold
    attrspace_id = check_H5Screate(H5S_SCALAR);
    buffer = UNBOUND_THRESHOLD;
    attr_id = check_H5Acreate(group_id, "UnboundThreshold", H5T_NATIVE_FLOAT, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_FLOAT, &buffer);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // FOFFraction
    attrspace_id = check_H5Screate(H5S_SCALAR);
    buffer = FOF_FRACTION;
    attr_id = check_H5Acreate(group_id, "FOFFraction", H5T_NATIVE_FLOAT, attrspace_id);
    check_H5Awrite(attr_id, H5T_NATIVE_FLOAT, &buffer);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // StrictSOMasses
    attrtype_id = check_H5Tcopy(H5T_C_S1);
    check_H5Tset_size(attrtype_id, 4);
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "StrictSOMasses", attrtype_id, attrspace_id);
    if (STRICT_SO_MASSES)
        check_H5Awrite(attr_id, attrtype_id, "Yes");
    else
        check_H5Awrite(attr_id, attrtype_id, "No");
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);
    check_H5Tclose(attrtype_id);

    // TotalNumberHalos
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "TotalNumberHalos", H5T_NATIVE_LLONG, attrspace_id);
    ibuffer = tot_num_halos;
    check_H5Awrite(attr_id, H5T_NATIVE_LLONG, &ibuffer);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // TotalChunks
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "TotalChunks", H5T_NATIVE_LLONG, attrspace_id);
    ibuffer = NUM_WRITERS;
    check_H5Awrite(attr_id, H5T_NATIVE_LLONG, &ibuffer);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    // TotalNumberParticles
    attrspace_id = check_H5Screate(H5S_SCALAR);
    attr_id = check_H5Acreate(group_id, "TotalNumberParticles", H5T_NATIVE_LLONG, attrspace_id);
    ibuffer = tot_num_p;
    check_H5Awrite(attr_id, H5T_NATIVE_LLONG, &ibuffer);
    check_H5Aclose(attr_id);
    check_H5Sclose(attrspace_id);

    check_H5Gclose(group_id);
}

void read_hdf5_header(hid_t HDF_FileID, struct binary_output_header *bh, char *filename) {
    hid_t attr_id, attrspace_id, attrtype_id;
    int64_t ndims;
    hsize_t dimsize;

    hid_t group_id = check_H5Gopen(HDF_FileID, "Header", filename);

    // Magic
    attr_id = check_H5Aopen(group_id, "Magic", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_ULLONG, &bh->magic);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // Snap
    attr_id = check_H5Aopen(group_id, "Snap", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_LLONG, &bh->snap);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // Chunk
    attr_id = check_H5Aopen(group_id, "Chunk", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_LLONG, &bh->chunk);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // ScaleFactor
    attr_id = check_H5Aopen(group_id, "ScaleFactor", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_FLOAT, &bh->scale);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // Omega_m
    attr_id = check_H5Aopen(group_id, "Omega_m", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_FLOAT, &bh->Om);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // Omega_L
    attr_id = check_H5Aopen(group_id, "Omega_L", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_FLOAT, &bh->Ol);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // H0
    attr_id = check_H5Aopen(group_id, "h", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_FLOAT, &bh->h0);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // Bounds
    attr_id = check_H5Aopen(group_id, "Bounds", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    ndims = check_H5Sget_simple_extent_ndims(attrspace_id);
    assert(ndims == 1);
    check_H5Sget_simple_extent_dims(attrspace_id, &dimsize);
    assert(dimsize == 6);
    check_H5Aread2(attr_id, H5T_NATIVE_FLOAT, &bh->bounds[0]);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // NumberHalos
    attr_id = check_H5Aopen(group_id, "NumberHalos", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_LLONG, &bh->num_halos);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // NumberParticles
    attr_id = check_H5Aopen(group_id, "NumberParticles", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_LLONG, &bh->num_particles);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // BoxSize
    attr_id = check_H5Aopen(group_id, "BoxSize", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_FLOAT, &bh->box_size);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // ParticleMass
    attr_id = check_H5Aopen(group_id, "ParticleMass", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_FLOAT, &bh->particle_mass);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // ParticleType
    attr_id = check_H5Aopen(group_id, "ParticleType", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_LLONG, &bh->particle_type);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // FormatRevision
    attr_id = check_H5Aopen(group_id, "FormatRevision", filename);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, H5T_NATIVE_INT, &bh->format_revision);
    check_H5Sclose(attrspace_id);
    check_H5Aclose(attr_id);

    // Version
    attr_id = check_H5Aopen(group_id, "Version", filename);
    attrtype_id = check_H5Tcopy(H5T_C_S1);
    check_H5Tset_size(attrtype_id, VERSION_MAX_SIZE);
    attrspace_id = check_H5Aget_space(attr_id);
    check_H5Sselect_all(attrspace_id);
    check_H5Aread2(attr_id, attrtype_id, &bh->rockstar_version[0]);
    check_H5Sclose(attrspace_id);
    check_H5Tclose(attrtype_id);
    check_H5Aclose(attr_id);

    check_H5Gclose(group_id);
}


void output_hdf5(int64_t id_offset, int64_t snap, int64_t chunk,
                 float *bounds, int64_t tot_num_halos, int64_t tot_num_p,
                 int64_t output_particles) {
    float                       max[3] = {0}, min[3] = {0};
    char                        filename[1024], dataid[256], description[256];
    int64_t                     i, j, num_write, num_particles;
    int64_t                     offset, id;
    int64_t                    *ids, *p_start;
    int                        *to_write; 
    struct binary_output_header bheader;
    hid_t                       HDF_FileID, HDF_FileID_Part, HDF_GroupID;
    hsize_t                     dims1[1], dims3[2], dims6[2], dims_part[1];
    float                      *buffer_float;
    int64_t                    *buffer_int, *buffer_id;


    memset(&bheader, 0, sizeof(struct binary_output_header));
    to_write = (int *) malloc(sizeof(int) * num_halos);
    ids = (int64_t *) malloc(sizeof(int64_t) * num_halos);
    p_start = (int64_t *) malloc(sizeof(int64_t) * num_halos);


    // Output Halos
    if (num_halos) {
        memcpy(min, halos[0].pos, sizeof(float) * 3);
        memcpy(max, halos[0].pos, sizeof(float) * 3);
    }
    for (i = 0, id = 0; i < num_halos; i++) {
        if (!_should_print(halos + i, bounds)) {
            to_write[i] = 0;
            continue;
        }

        to_write[i] = 1;

        for (j = 0; j < 3; j++) {
            if (min[j] > halos[i].pos[j])
                min[j] = halos[i].pos[j];
            if (max[j] < halos[i].pos[j])
                max[j] = halos[i].pos[j];
        }

        ids[id] = id + id_offset;
        p_start[id] = bheader.num_particles;
        bheader.num_halos++;
        bheader.num_particles += halos[i].num_p;
        id++;
    }

    num_write = bheader.num_halos;
    num_particles = bheader.num_particles;

    // set data dimensions and buffers
    dims1[0] = num_write;

    dims3[0] = num_write;
    dims3[1] = 3;

    dims6[0] = num_write;
    dims6[1] = 6;

    buffer_float = (float *) malloc(sizeof(float) * 6 * num_write);
    buffer_int = (int64_t *) malloc(sizeof(int64_t) * num_write);


    // Create file
    get_output_filename(filename, 1024, snap, chunk, "hdf5");
    HDF_FileID = check_H5Fcreate(filename, H5F_ACC_TRUNC);


    HDF_GroupID = check_H5Gcreate(HDF_FileID, "/Subhalo");

    // ID and ParticleStart are already set, write these data here
    write_hdf5_dataset(HDF_GroupID, "ID", H5T_NATIVE_LLONG, 1, dims1, ids);
    add_hdf5_attribute(HDF_GroupID, "ID", "N/A",
                       "Halo ID");
    free(ids);

    write_hdf5_dataset(HDF_GroupID, "ParticleStart", H5T_NATIVE_LLONG, 1, dims1, p_start);
    add_hdf5_attribute(HDF_GroupID, "ParticleStart", "N/A",
                       "Index of the first particle in the halo");
    free(p_start);

    // Write data to hdf5 file
    set_buffer(buffer_float, to_write, (char *) &(halos[0].pos[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "Position", H5T_NATIVE_FLOAT, 2, dims3, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "Position", "Mpc/h (comoving)",
                       "Position of the halo centre (potential minimum)");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].pos[3]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "Velocity", H5T_NATIVE_FLOAT, 2, dims3, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "Velocity", "km/s (physical, peculiar)",
                       "Velocity of the halo");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].corevel[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "CoreVelocity", H5T_NATIVE_FLOAT, 2, dims3, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "CoreVelocity", "km/s (physical, peculiar)",
                       "Velocity of core");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].bulkvel[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "BulkVelocity", H5T_NATIVE_FLOAT, 2, dims3, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "BulkVelocity", "km/s (physical, peculiar)",
                       "Bulk velocity");

    sprintf(dataid, "M%s", MASS_DEFINITION);
    sprintf(description, "Halo mass (%s)", MASS_DEFINITION);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].m) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, dataid, H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, dataid, "Msun/h",
                       description);

    sprintf(dataid, "R%s", MASS_DEFINITION);
    sprintf(description, "Halo radius (%s)", MASS_DEFINITION);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].r) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, dataid, H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, dataid, "kpc / h (comoving)",
                       description);

    set_buffer(buffer_float, to_write, (char *) &(halos[0].child_r) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "ChildRadius", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "ChildRadius", "kpc/h (comoving)",
                       "Child radius");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].vmax_r) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "RadiusDynamicalVirial", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "RadiusDynamicalVirial", "kpc/h (comoving)",
                       "Radius of Vmax times dynamical time, used to normalize the position-space distance");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].mgrav) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "Mbound", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "Mbound", "Msun/h",
                       "All bounded mass");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].vmax) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "Vmax", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "Vmax", "km/s (physical, peculiar)",
                       "Maximum circular velocity");

#ifdef OUTPUT_RVMAX
    set_buffer(buffer_float, to_write, (char *) &(halos[0].rvmax) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "VmaxRadius", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "VmaxRadius", "kpc/h (comoving)",
                       "Radius where circular velocity is maximum");
#endif

#ifdef OUTPUT_NFW_CHI2
    set_buffer(buffer_float, to_write, (char *) &(halos[0].chi2) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "NFWChi2", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "NFWChi2", "N/A",
                       "Chi square of the NFW fitting");
#endif

    set_buffer(buffer_float, to_write, (char *) &(halos[0].rs) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "ScaleRadius", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "ScaleRadius", "kpc/h (comoving)",
                       "Scale radius");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].klypin_rs) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "ScaleRadiusKlypin", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "ScaleRadiusKlypin", "kpc/h (comoving)",
                       "Scale radius in Klypin's definition");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].vrms) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "Vrms", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "Vrms", "km/s (physical, peculiar)",
                       "Root-mean square velocity dispersion");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].J[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "AngularMomentum", H5T_NATIVE_FLOAT, 2, dims3, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "AngularMomentum", "(Msun/h) * (Mpc/h) * km/s (physical)",
                       "Angular momentum");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].energy) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "Energy", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "Energy", "(Msun/h) * (km/s)^2 (physical)",
                       "Energy (kinetic + potential)");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].spin) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "Spin", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "Spin", "Dimensionless",
                       "Halo spin parameter (Peebles)");

    sprintf(dataid, "M%s", MASS_DEFINITION2);
    sprintf(description, "Halo mass (%s)", MASS_DEFINITION2);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].alt_m[0]) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, dataid, H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, dataid, "Msun/h",
                       description);

    sprintf(dataid, "M%s", MASS_DEFINITION3);
    sprintf(description, "Halo mass (%s)", MASS_DEFINITION3);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].alt_m[1]) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, dataid, H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, dataid, "Msun/h",
                       description);

    sprintf(dataid, "M%s", MASS_DEFINITION4);
    sprintf(description, "Halo mass (%s)", MASS_DEFINITION4);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].alt_m[2]) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, dataid, H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, dataid, "Msun/h",
                       description);

    sprintf(dataid, "M%s", MASS_DEFINITION5);
    sprintf(description, "Halo mass (%s)", MASS_DEFINITION5);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].alt_m[3]) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, dataid, H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, dataid, "Msun/h",
                       description);

    set_buffer(buffer_float, to_write, (char *) &(halos[0].Xoff) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "PositionOffset", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "PositionOffset", "kpc/h (comoving)",
                       "Position offset (mean - potential minimum)");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].Voff) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "VelocityOffset", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "VelocityOffset", "km/s (physical, peculiar)",
                       "Velocity offset (mean - potential minimum)");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].bullock_spin) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "SpinBullock", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "SpinBullock", "Dimensionless",
                       "Halo spin parameter (Bullock)");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].kin_to_pot) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "KineticToPotentialRatio", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "KineticToPotentialRatio", "Dimensionless",
                       "Ratio between kinetic and potential energies: T/|U|");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].m_pe_b) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "MpeBehroozi", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "MpeBehroozi", "Msun/h",
                       "Pseudo-evolution corrected mass (Behroozi)");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].m_pe_d) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "MpeDiemer", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "MpeDiemer", "Msun/h",
                       "Pseudo-evolution corrected mass (Diemer)");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].halfmass_radius) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "HalfMassRadius", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "HalfMassRadius", "kpc/h (comoving)",
                       "Half-mass radius");

    set_buffer(buffer_int, to_write, (char *) &(halos[0].num_p) - (char *) (halos), 1, H5T_NATIVE_LLONG);
    write_hdf5_dataset(HDF_GroupID, "NumberParticles", H5T_NATIVE_LLONG, 1, dims1, buffer_int);
    add_hdf5_attribute(HDF_GroupID, "NumberParticles", "N/A",
                       "Number of particles");

    set_buffer(buffer_int, to_write, (char *) &(halos[0].num_child_particles) - (char *) (halos), 1, H5T_NATIVE_LLONG);
    write_hdf5_dataset(HDF_GroupID, "NumberChildParticles", H5T_NATIVE_LLONG, 1, dims1, buffer_int);
    add_hdf5_attribute(HDF_GroupID, "NumberChildParticles", "N/A",
                       "Number of child particles");

    set_buffer(buffer_int, to_write, (char *) &(halos[0].desc) - (char *) (halos), 1, H5T_NATIVE_LLONG);
    write_hdf5_dataset(HDF_GroupID, "DescendantID", H5T_NATIVE_LLONG, 1, dims1, buffer_int);
    add_hdf5_attribute(HDF_GroupID, "DescendantID", "N/A",
                       "ID of the descendant halo");

    set_buffer(buffer_int, to_write, (char *) &(halos[0].flags) - (char *) (halos), 1, H5T_NATIVE_LLONG);
    write_hdf5_dataset(HDF_GroupID, "Flags", H5T_NATIVE_LLONG, 1, dims1, buffer_int);
    add_hdf5_attribute(HDF_GroupID, "Flags", "N/A",
                       "Halo flags");

    set_buffer(buffer_int, to_write, (char *) &(halos[0].n_core) - (char *) (halos), 1, H5T_NATIVE_LLONG);
    write_hdf5_dataset(HDF_GroupID, "NumberCoreParticles", H5T_NATIVE_LLONG, 1, dims1, buffer_int);
    add_hdf5_attribute(HDF_GroupID, "NumberCoreParticles", "N/A",
                       "Number of core particles");

    check_H5Gclose(HDF_GroupID);

    HDF_GroupID = check_H5Gcreate(HDF_FileID, "/Shape");

    sprintf(description, "Intermediate to major axis ratio (mass definition: %s): b/a", MASS_DEFINITION);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].b_to_a) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "IntermediateToMajorAxisRatio", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "IntermediateToMajorAxisRatio", "Dimensionless",
                       description);

    sprintf(description, "Minor to major axis ratio (mass definition: %s): c/a", MASS_DEFINITION);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].c_to_a) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "MinorToMajorAxisRatio", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "MinorToMajorAxisRatio", "Dimensionless",
                       description);

    sprintf(description, "Intermediate to major axis ratio (mass definition: %s): b/a", MASS_DEFINITION4);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].b_to_a2) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "IntermediateToMajorAxisRatio2", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "IntermediateToMajorAxisRatio2", "Dimensionless",
                       description);

    sprintf(description, "Minor to major axis ratio (mass definition: %s): c/a", MASS_DEFINITION4);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].c_to_a2) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "MinorToMajorAxisRatio2", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "MinorToMajorAxisRatio2", "Dimensionless",
                       description);

    sprintf(description, "Major axis direction (mass definition: %s)", MASS_DEFINITION);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].A[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "MajorAxis", H5T_NATIVE_FLOAT, 2, dims3, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "MajorAxis", "Normalized",
                       description);

    sprintf(description, "Major axis direction (mass definition: %s)", MASS_DEFINITION4);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].A2[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "MajorAxis2", H5T_NATIVE_FLOAT, 2, dims3, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "MajorAxis2", "Normalized",
                       description);

#ifdef OUTPUT_INTERMEDIATE_AXIS
    sprintf(description, "Intermediate axis direction (mass definition: %s)", MASS_DEFINITION);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].A_I[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "IntermediateAxis", H5T_NATIVE_FLOAT, 2, dims3, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "IntermediateAxis", "Normalized",
                       description);

    sprintf(description, "Intermediate axis direction (mass definition: %s)", MASS_DEFINITION4);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].A2_I[0]) - (char *) (halos), 3, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "IntermediateAxis2", H5T_NATIVE_FLOAT, 2, dims3, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "IntermediateAxis2", "Normalized",
                       description);
#endif

#ifdef OUTPUT_INERTIA_TENSOR
    sprintf(description, "Inertia tensor (mass definition: %s): (Ixx, Iyy, Izz, Ixy, Iyz, Izx)", MASS_DEFINITION);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].inertia_tensor[0]) - (char *) (halos), 6, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "InertiaTensor", H5T_NATIVE_FLOAT, 2, dims6, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "InertiaTensor", "Msun/h * (kpc/h)^2 (comoving)",
                       description);

    sprintf(description, "Inertia tensor (mass definition: %s): (Ixx, Iyy, Izz, Ixy, Iyz, Izx)", MASS_DEFINITION4);
    set_buffer(buffer_float, to_write, (char *) &(halos[0].inertia_tensor2[0]) - (char *) (halos), 6, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "InertiaTensor2", H5T_NATIVE_FLOAT, 2, dims6, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "InertiaTensor2", "Msun/h * (kpc/h)^2 (comoving)",
                       description);
#endif

    check_H5Gclose(HDF_GroupID);

    HDF_GroupID = check_H5Gcreate(HDF_FileID, "/Error");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].min_pos_err) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "MinPosError", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "MinPosError", "Mpc/h (comoving)",
                       "Uncertainty of position");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].min_vel_err) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "MinVelError", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "MinVelError", "km/s (physical, peculiar)",
                       "Uncertainty of velocity");

    set_buffer(buffer_float, to_write, (char *) &(halos[0].min_bulkvel_err) - (char *) (halos), 1, H5T_NATIVE_FLOAT);
    write_hdf5_dataset(HDF_GroupID, "MinBulkVelError", H5T_NATIVE_FLOAT, 1, dims1, buffer_float);
    add_hdf5_attribute(HDF_GroupID, "MinBulkVelError", "km/s (physical, peculiar)",
                       "Uncertainty of bulk velocity");

    check_H5Gclose(HDF_GroupID);

    free(buffer_float);
    free(buffer_int);

    // Output header
    fill_binary_header(&bheader, snap, chunk);
    bheader.particle_type = PARTICLE_TYPE_IDS;
    if (bounds)
        memcpy(bheader.bounds, bounds, sizeof(float) * 6);
    else {
        memcpy(bheader.bounds, min, sizeof(float) * 3);
        memcpy(&(bheader.bounds[3]), max, sizeof(float) * 3);
    }
    write_hdf5_header(HDF_FileID, &bheader, tot_num_halos, tot_num_p);


    // Output Particles
    if (output_particles) {
        get_output_filename(filename, 1024, snap, chunk, "id.hdf5");
        HDF_FileID_Part = check_H5Fcreate(filename, H5F_ACC_TRUNC);

        buffer_id = (int64_t *) malloc(sizeof(int64_t) * num_particles);
        dims_part[0] = num_particles;
        for (i = 0, offset = 0; i < num_halos; i++) {
            if (!to_write[i])
                continue;
            for (j = 0; j < halos[i].num_p; j++)
                buffer_id[offset+j] = p[halos[i].p_start + j].id;
            offset += halos[i].num_p;
        }
        write_hdf5_dataset(HDF_FileID_Part, "/ParticleID",
                           H5T_NATIVE_LLONG, 1, dims_part, buffer_id);
        add_hdf5_attribute(HDF_FileID_Part, "/ParticleID", "N/A",
                           "ID of particles");
        free(buffer_id);

        write_hdf5_header(HDF_FileID_Part, &bheader, tot_num_halos, tot_num_p);

        check_H5Fclose(HDF_FileID_Part);
    }


    free(to_write);
    check_H5Fclose(HDF_FileID);
}

void load_hdf5_header(int64_t snap, int64_t chunk,
                      struct binary_output_header *bheader) {
    char  filename[1024];
    hid_t HDF_FileID;

    get_output_filename(filename, 1024, snap, chunk, "hdf5");
    HDF_FileID = check_H5Fopen(filename, H5F_ACC_RDONLY);

    read_hdf5_header(HDF_FileID, bheader, filename);

    assert(bheader->magic == ROCKSTAR_MAGIC);
    check_H5Fclose(HDF_FileID);
}

void load_hdf5_halos(int64_t snap, int64_t chunk,
                     struct binary_output_header *bheader,
                     struct halo **halos, int64_t **part_ids,
                     int64_t coalesced) {
    char  filename[1024];
    hid_t HDF_FileID, HDF_FileID_Part;

    if (!coalesced)
        get_output_filename(filename, 1024, snap, chunk, "hdf5");
    else
        get_output_filename(filename, 1024, snap, chunk, "coalesced.hdf5");
    HDF_FileID = check_H5Fopen(filename, H5F_ACC_RDONLY);
    read_hdf5_header(HDF_FileID, bheader, filename);
    assert(bheader->magic == ROCKSTAR_MAGIC);
    assert(bheader->num_halos >= 0);
    check_realloc_s(*halos, sizeof(struct halo), bheader->num_halos);
    read_hdf5_halos(HDF_FileID, *halos, bheader);
    check_H5Fclose(HDF_FileID);

    get_output_filename(filename, 1024, snap, chunk, "id.hdf5");
    HDF_FileID_Part = check_H5Fopen(filename, H5F_ACC_RDONLY);
    read_hdf5_header(HDF_FileID_Part, bheader, filename);
    assert(bheader->magic == ROCKSTAR_MAGIC);
    assert(bheader->num_particles >= 0);
    check_realloc_s(*part_ids, sizeof(int64_t), bheader->num_particles);
    read_hdf5_dataset(HDF_FileID_Part, "/ParticleID", H5T_NATIVE_LLONG, *part_ids);
    check_H5Fclose(HDF_FileID_Part);

    /*
    if (i != bheader->num_halos || j != bheader->num_particles) {
        fprintf(stderr, "[Error] Truncated input file %s!\n", buffer);
        exit(1);
    }
    */

    read_binary_header_config(bheader);
}

#endif /* ENABLE_HDF5 */
