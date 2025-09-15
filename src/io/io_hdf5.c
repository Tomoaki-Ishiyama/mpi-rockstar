#ifdef ENABLE_HDF5
#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h> /* HDF5 required */
#include <inttypes.h>
#include "../error.h"

hid_t check_H5Fopen(char *filename, unsigned flags) {
    hid_t HDF_FileID = H5Fopen(filename, flags, H5P_DEFAULT);
    if (HDF_FileID < 0) {
        fprintf(stderr, "[Error] Failed to open HDF5 file %s!\n", filename);
        exit(1);
    }
    return HDF_FileID;
}

hid_t check_H5Gopen(hid_t HDF_FileID, char *gid, char *filename) {
    hid_t HDF_GroupID = H5Gopen(HDF_FileID, gid);
    if (HDF_GroupID < 0) {
        fprintf(stderr, "[Error] Failed to open group %s in HDF5 file %s!\n",
                gid, filename);
        exit(1);
    }
    return HDF_GroupID;
}

hid_t check_H5Gopen2(hid_t HDF_FileID, char *gid) {
    hid_t HDF_GroupID = H5Gopen(HDF_FileID, gid);
    if (HDF_GroupID < 0) {
        fprintf(stderr, "[Error] Failed to open group %s !\n", gid);
        exit(1);
    }
    return HDF_GroupID;
}

hid_t check_H5Dopen(hid_t HDF_GroupID, char *dataid, char *gid,
                    char *filename) {
    hid_t HDF_DatasetID = H5Dopen(HDF_GroupID, dataid);
    if (HDF_DatasetID < 0) {
        fprintf(stderr,
                "[Error] Failed to open dataset %s/%s in HDF5 file %s!\n", gid,
                dataid, filename);
        exit(1);
    }
    return HDF_DatasetID;
}

hid_t check_H5Dopen2(hid_t HDF_GroupID, char *dataid) {
    hid_t HDF_DatasetID = H5Dopen(HDF_GroupID, dataid);
    if (HDF_DatasetID < 0) {
        fprintf(stderr, "[Error] Failed to open dataset %s!\n", dataid);
        exit(1);
    }
    return HDF_DatasetID;
}

hid_t check_H5Dget_space(hid_t HDF_DatasetID) {
    hid_t HDF_DataspaceID = H5Dget_space(HDF_DatasetID);
    if (HDF_DataspaceID < 0) {
        fprintf(stderr, "[Error] Failed to get HDF5 dataspace!\n");
        exit(1);
    }
    return HDF_DataspaceID;
}

void check_H5Dread(hid_t HDF_DatasetID, hid_t type, void *buffer, char *dataid,
                   char *gid, char *filename) {
    if (H5Dread(HDF_DatasetID, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) <
        0) {
        fprintf(stderr,
                "[Error] failed to read dataset %s/%s in HDF5 file %s\n", gid,
                dataid, filename);
        exit(1);
    }
}

void check_H5Dread2(hid_t HDF_DatasetID, hid_t type, void *buffer, char *dataid) {
    if (H5Dread(HDF_DatasetID, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) <
        0) {
        fprintf(stderr,
                "[Error] failed to read dataset %s!\n", dataid);
        exit(1);
    }
}

hid_t check_H5Aopen_name(hid_t HDF_GroupID, char *dataid, char *gid,
                         char *filename) {
    hid_t HDF_AttrID = H5Aopen_name(HDF_GroupID, dataid);
    if (HDF_AttrID < 0) {
        fprintf(stderr,
                "[Error] Failed to open attribute %s/%s in HDF5 file %s!\n",
                gid, dataid, filename);
        exit(1);
    }
    return HDF_AttrID;
}

hid_t check_H5Aopen(hid_t HDF_GroupID, char *dataid, char *filename) {
    hid_t HDF_AttrID = H5Aopen(HDF_GroupID, dataid, H5P_DEFAULT);
    if (HDF_AttrID < 0) {
        fprintf(stderr,
                "[Error] Failed to open attribute %s in HDF5 file %s!\n",
                dataid, filename);
        exit(1);
    }
    return HDF_AttrID;
}

hid_t check_H5Aget_space(hid_t HDF_AttrID) {
    hid_t HDF_DataspaceID = H5Aget_space(HDF_AttrID);
    if (HDF_AttrID < 0) {
        fprintf(stderr, "[Error] Failed to get HDF5 dataspace!\n");
        exit(1);
    }
    return HDF_DataspaceID;
}

void check_H5Aread(hid_t HDF_AttrID, hid_t type, void *buffer, char *dataid,
                   char *gid, char *filename) {
    if (H5Aread(HDF_AttrID, type, buffer) < 0) {
        fprintf(stderr,
                "[Error] failed to read attribute %s/%s in HDF5 file %s\n", gid,
                dataid, filename);
        exit(1);
    }
}

void check_H5Aread2(hid_t HDF_AttrID, hid_t type, void *buffer) {
    if (H5Aread(HDF_AttrID, type, buffer) < 0) {
        fprintf(stderr, "[Error] failed to read attribute\n");
        exit(1);
    }
}

void check_H5Sselect_all(hid_t HDF_DataspaceID) {
    if (H5Sselect_all(HDF_DataspaceID) < 0 ||
        H5Sselect_valid(HDF_DataspaceID) <= 0) {
        fprintf(stderr,
                "[Error] Failed to select all elements in HDF5 dataspace!\n");
        exit(1);
    }
}

int64_t check_H5Sget_simple_extent_ndims(hid_t HDF_DataspaceID) {
    int64_t ndims = H5Sget_simple_extent_ndims(HDF_DataspaceID);
    if (ndims < 0) {
        fprintf(
            stderr,
            "[Error] Failed to get number of dimensions for HDF5 dataspace!\n");
        exit(1);
    }
    return ndims;
}

void check_H5Sget_simple_extent_dims(hid_t HDF_DataspaceID, hsize_t *dimsize) {
    if (H5Sget_simple_extent_dims(HDF_DataspaceID, dimsize, NULL) < 0) {
        fprintf(stderr,
                "[Error] Failed to get dimensions for HDF5 dataspace!\n");
        exit(1);
    }
}

hid_t check_H5Fcreate(char *filename, unsigned flags) {
    hid_t HDF_FileID = H5Fcreate(filename, flags, H5P_DEFAULT, H5P_DEFAULT);
    if (HDF_FileID < 0) {
        fprintf(stderr, "[Error] Failed to create HDF5 file %s!\n", filename);
        exit(1);
    }
    return HDF_FileID;
}

void check_H5Fclose(hid_t HDF_FileID) {
    if (H5Fclose(HDF_FileID) < 0) {
        fprintf(stderr, "[Error] Failed to close HDF5 file!\n");
        exit(1);
    }
}

hid_t check_H5Screate_simple(hsize_t rank, hsize_t *dims, hsize_t *maxdims) {
    hid_t HDF_DataspaceID = H5Screate_simple(rank, dims, maxdims);
    if (HDF_DataspaceID < 0) {
        fprintf(stderr, "[Error] Failed to create HDF5 dataspace!\n");
        exit(1);
    }
    return HDF_DataspaceID;
}

hid_t check_H5Screate(H5S_class_t type) {
    hid_t HDF_DataspaceID = H5Screate(type);
    if (HDF_DataspaceID < 0) {
        fprintf(stderr, "[Error] Failed to create HDF5 dataspace!\n");
        exit(1);
    }
    return HDF_DataspaceID;
}

void check_H5Sset_extent_simple(hid_t HDF_DataspaceID, hsize_t rank,
                                hsize_t *dims, hsize_t *maxdims) {
    if (H5Sset_extent_simple(HDF_DataspaceID, rank, dims, maxdims) < 0) {
        fprintf(stderr, "[Error] Failed to set dataspace!\n");
        exit(1);
    }
}

void check_H5Sclose(hid_t HDF_DataspaceID) {
    if (H5Sclose(HDF_DataspaceID) < 0) {
        fprintf(stderr, "[Error] Failed to close dataspace!\n");
        exit(1);
    }
}

hid_t check_H5Tcopy(hid_t type) {
    hid_t HDF_TypeID = H5Tcopy(type);
    if (HDF_TypeID < 0) {
        fprintf(stderr, "[Error] Failed to copy HDF5 datatype!\n");
        exit(1);
    }
    return HDF_TypeID;
}

void check_H5Tset_size(hid_t type, size_t size) {
    if (H5Tset_size(type, size) < 0) {
        fprintf(stderr, "[Error] Failed to set HDF5 datatype!\n");
        exit(1);
    }
}

void check_H5Tclose(hid_t HDF_TypeID) {
    if (H5Tclose(HDF_TypeID) < 0) {
        fprintf(stderr, "[Error] Failed to close datatype!\n");
        exit(1);
    }
}

hid_t check_H5Dcreate(hid_t HDF_GroupID, char *dataid, hid_t type,
                      hid_t HDF_DataspaceID) {
    hid_t HDF_DatasetID = H5Dcreate2(HDF_GroupID, dataid, type,
                                     HDF_DataspaceID, H5P_DEFAULT,
                                     H5P_DEFAULT, H5P_DEFAULT);
    if (HDF_DatasetID < 0) {
        fprintf(stderr, "[Error] Failed to create dataset %s!\n", dataid);
        exit(1);
    }
    return HDF_DatasetID;
}

void check_H5Dwrite(hid_t HDF_DatasetID, hid_t type, void *buffer) {
    if (H5Dwrite(HDF_DatasetID, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) < 0) {
        fprintf(stderr, "[Error] Failed to write HDF5 dataset!\n");
        exit(1);
    }
}

void check_H5Dclose(hid_t HDF_DatasetID) {
    if (H5Dclose(HDF_DatasetID) < 0) {
        fprintf(stderr, "[Error] Failed to close dataset!\n");
        exit(1);
    }
}

hid_t check_H5Acreate(hid_t HDF_DatasetID, char *dataid, hid_t type,
                      hid_t HDF_DataspaceID) {
    hid_t HDF_AttrID = H5Acreate2(HDF_DatasetID, dataid, type, HDF_DataspaceID,
                                  H5P_DEFAULT, H5P_DEFAULT);
    if (HDF_AttrID < 0) {
        fprintf(stderr, "[Error] Failed to create attribute %s!\n", dataid);
        exit(1);
    }
    return HDF_AttrID;
}

void check_H5Awrite(hid_t HDF_AttrID, hid_t type, void *buffer) {
    if (H5Awrite(HDF_AttrID, type, buffer) < 0) {
        fprintf(stderr, "[Error] Failed to write HDF5 attribute!\n");
        exit(1);
    }
}

void check_H5Aclose(hid_t HDF_AttrID) {
    if (H5Aclose(HDF_AttrID) < 0) {
        fprintf(stderr, "[Error] Failed to close attribute!\n");
        exit(1);
    }
}

hid_t check_H5Gcreate(hid_t HDF_FileID, char *groupid) {
    hid_t HDF_GroupID = H5Gcreate2(HDF_FileID, groupid,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (HDF_GroupID < 0) {
        fprintf(stderr, "[Error] Failed to create group %s!\n", groupid);
        exit(1);
    }
    return HDF_GroupID;
}

void check_H5Gclose(hid_t HDF_GroupID) {
    if (H5Gclose(HDF_GroupID) < 0) {
        fprintf(stderr, "[Error] Failed to close group!\n");
        exit(1);
    }
}

#endif /* ENABLE_HDF5 */
