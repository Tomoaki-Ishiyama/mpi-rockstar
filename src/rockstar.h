#ifndef ROCKSTAR_H
#define ROCKSTAR_H

#include <stdint.h>
#include "particle.h"
#include "halo.h"
#include "fof.h"
#include "error.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288 /* From math.h */
#endif                                             /* M_PI */

#define MIN_WORKUNIT    5000000
#define LARGE_FOF       (MIN_WORKUNIT / sizeof(struct particle))
#define FOF_SKIP_THRESH 16

extern struct particle  *p;
extern struct bparticle *bp;
extern int64_t           num_p, num_bp, num_additional_p;
extern int64_t           num_all_fofs;
extern int64_t           num_bfofs;
extern struct fof       *all_fofs;

struct workunit_info {
    int64_t num_fofs, num_halos, num_particles, chunk;
    int64_t num_meta_fofs, num_meta_p, total_bg;
    float   bounds[6];
};

#include "interleaving.h"

void              rockstar(float *bounds, int64_t manual_subs);
void              rockstar_cleanup();
void              prune_fofs(float *bounds);
void              build_particle_tree(void);
void              clear_particle_tree(void);
struct particle **find_halo_sphere(struct halo *h, int64_t *num_results);
int               sort_fofs(const void *a, const void *b);
void              convert_bgroups_to_metafofs(void);
void              do_workunit(struct workunit_info *w, struct fof *fofs);
void find_unfinished_workunit(struct workunit_info *w, struct fof **fofs,
                              struct particle **parts, int64_t **set_sizes,
                              struct bgroup **bgroup_list);
void integrate_finished_workunit(struct workunit_info *w, struct fof *fofs,
                                 struct halo *h, struct extra_halo_info *ei,
                                 struct particle *parts);
void fof_of_id(int64_t id, struct fof *tf);
void particle_cleanup();

void align_particles(struct fof f);
void realloc_halos(int64_t new_num_halos);
void copy_halos(const struct HaloInfo *haloinfo, int64_t offset);

#endif /* ROCKSTAR_H */
