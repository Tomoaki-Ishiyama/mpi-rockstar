#ifndef GROUPIES_H
#define GROUPIES_H

#include "fof.h"
#include "halo.h"
#include <stdint.h>

#define GROWING_FLAG       1
#define DELETE_FLAG        2
#define POSSIBLE_SWAP_FLAG 4
#define TAGGED_FLAG        8
#define ALWAYS_PRINT_FLAG  16

extern struct halo            *halos;
extern int64_t                 num_halos;
extern struct extra_halo_info *extra_info;

extern double particle_thresh_dens[5];
extern double  particle_rvir_dens , particle_rvir_dens_z0;
extern int64_t min_dens_index;
extern double  dynamical_time;
extern float   scale_dx;
#pragma omp threadprivate(particle_thresh_dens)
#pragma omp threadprivate(particle_rvir_dens, particle_rvir_dens_z0)
#pragma omp threadprivate(min_dens_index, dynamical_time, scale_dx)

struct HaloInfo {
    struct halo            *halos;
    struct extra_halo_info *extra_info;
    int64_t                 num_halos;
};

void init_haloinfo(struct HaloInfo *haloinfo);
void free_haloinfo(struct HaloInfo *haloinfo);

void  find_subs(struct fof *f, struct FOFInfo *fofinfo,
                struct HaloInfo *haloinfo);
void  calc_mass_definition(void);
void  free_particle_copies(void);
void  free_halos(void);
float max_halo_radius(struct halo *h);

// Internal functions
void  add_new_halo(struct HaloInfo *haloinfo);
void  alloc_particle_copies(int64_t total_copies);
void  norm_sd(struct fof *f, float thresh);
float find_median_r(float *rad, int64_t num_p, float frac, unsigned int *seedp);
float random_unit(void);

void find_sd(struct particle *particles, int64_t n, double *sig_x,
             double *sig_v);
#endif /* GROUPIES_H */
