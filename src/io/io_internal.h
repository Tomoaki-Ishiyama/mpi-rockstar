#ifndef _IO_INTERNAL_H_
#define _IO_INTERNAL_H_
#include <inttypes.h>
#include "../particle.h"

#define ROCKSTAR_MAGIC     (uint64_t)0xfadedacec0c0d0d0
#define BINARY_HEADER_SIZE 256
#define PARTICLE_TYPE_NONE 0
#define PARTICLE_TYPE_IDS  1
#define PARTICLE_TYPE_FULL 2

#define OUTPUT_BUFFER_SIZE 1000000
#define VERSION_MAX_SIZE   12

struct binary_output_header {
    uint64_t magic;
    int64_t  snap, chunk;
    float    scale, Om, Ol, h0;
    float    bounds[6];
    int64_t  num_halos, num_particles;
    float    box_size, particle_mass;
    int64_t  particle_type;
    int32_t  format_revision;
    char     rockstar_version[VERSION_MAX_SIZE];
    int32_t  add_flag;
    char     unused[BINARY_HEADER_SIZE - (sizeof(char) * VERSION_MAX_SIZE) -
                (sizeof(float) * 12) - (sizeof(int32_t) * 2) - (sizeof(int64_t) * 6)];
};

void fill_binary_header(struct binary_output_header *bh, int64_t snap,
                        int64_t chunk);

void read_binary_header_config(struct binary_output_header *bh);
void output_particles_internal(int64_t snap, int64_t chunk, double fraction);
void load_particles_internal(char *filename, struct particle **part,
                             int64_t *num_part);
void output_binary(int64_t id_offset, int64_t snap, int64_t chunk,
                   float *bounds, int64_t output_particles);
void load_binary_header(int64_t snap, int64_t chunk,
                        struct binary_output_header *bheader);
void load_binary_halos(int64_t snap, int64_t chunk,
                       struct binary_output_header *bheader,
                       struct halo **halos, int64_t **part_ids,
                       int64_t coalesced);
#endif /* _IO_INTERNAL_H_ */
