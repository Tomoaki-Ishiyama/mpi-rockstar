#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <inttypes.h>
#include <assert.h>
#include "rockstar.h"
#include "halo.h"
#include "fof.h"
#include "particle.h"
#include "groupies.h"
#include "subhalo_metric.h"
#include "check_syscalls.h"
#include "config_vars.h"
#include "universal_constants.h"
#include "potential.h"
#include "nfw.h"
#include "distance.h"
#include "fun_times.h"
#include "jacobi.h"
#include "hubble.h"

#define FAST3TREE_DIM    6
#define POINTS_PER_LEAF  40
#define FAST3TREE_PREFIX GROUPIES
#define FAST3TREE_TYPE   struct particle
#include "fast3tree.c"

struct particle  *copies         = NULL; // For storing phase-space FOFs
int64_t          *particle_halos = NULL;
float            *particle_r     = NULL;
struct potential *po             = NULL;
int64_t           num_alloc_pc = 0, num_copies = 0;

struct fof *subfofs     = NULL;
int64_t     num_subfofs = 0, num_alloced_subfofs = 0;

int64_t                 num_halos  = 0;
struct halo            *halos      = NULL;
struct extra_halo_info *extra_info = NULL;

struct fast3tree_results *res       = NULL;
struct fast3tree         *phasetree = NULL;

int64_t       num_alloc_gh = 0, num_growing_halos = 0;
struct halo **growing_halos = NULL;

int64_t *halo_ids             = NULL;
int64_t  num_alloced_halo_ids = 0;

#pragma omp threadprivate(copies, particle_halos, particle_r, po)
#pragma omp threadprivate(num_copies, num_alloc_pc)
#pragma omp threadprivate(subfofs, num_subfofs, num_alloced_subfofs)
#pragma omp threadprivate(res, phasetree)
#pragma omp threadprivate(growing_halos, num_growing_halos, num_alloc_gh)
#pragma omp threadprivate(halo_ids, num_alloced_halo_ids)

double  particle_thresh_dens[5] = {0};
double  particle_rvir_dens = 0, particle_rvir_dens_z0 = 0;
int64_t min_dens_index       = 0;
double  dynamical_time       = 0;
float   scale_dx             = 1;

void init_haloinfo(struct HaloInfo *haloinfo) {
    haloinfo->halos      = NULL;
    haloinfo->extra_info = NULL;
    haloinfo->num_halos  = 0;
}

void free_haloinfo(struct HaloInfo *haloinfo) {
    check_realloc_s(haloinfo->halos, 0, 0);
    check_realloc_s(haloinfo->extra_info, 0, 0);
    init_haloinfo(haloinfo);

    free_particle_copies();

    if (num_alloced_subfofs) {
        check_realloc_s(subfofs, 0, 0);
        num_alloced_subfofs = 0;
    }

    fast3tree_results_free(res);
    res = NULL;
    fast3tree_free(&phasetree);

    if (num_alloc_gh) {
        check_realloc_s(growing_halos, 0, 0);
        num_alloc_gh = 0;
    }
    if (num_alloced_halo_ids) {
        check_realloc_s(halo_ids, 0, 0);
        num_alloced_halo_ids = 0;
    }
}

double vir_density(double a) {
    double x = (Om / pow(a, 3)) / pow(hubble_scaling(1.0 / a - 1.0), 2.0) - 1.0;
    return ((18 * M_PI * M_PI + 82.0 * x - 39 * x * x) / (1.0 + x));
}

float _calc_mass_definition(char **md) {
    float   scale_now       = LIGHTCONE ? scale_dx : SCALE_NOW;
    int64_t length          = strlen(*md);
    char    last_char       = (length) ? md[0][length - 1] : 0;
    float   matter_fraction = (Om / pow(scale_now, 3)) /
                            pow(hubble_scaling(1.0 / scale_now - 1.0), 2.0);
    float cons = Om * CRITICAL_DENSITY / PARTICLE_MASS; // background density
    char *mass = *md;
    float thresh_dens;
    if (mass[0] == 'm' || mass[0] == 'M')
        mass++;

    if (last_char == 'b' || last_char == 'B')
        thresh_dens = atof(mass) * cons;
    else if (last_char == 'c' || last_char == 'C')
        thresh_dens = atof(mass) * cons / matter_fraction;
    else {
        if (strcasecmp(*md, "vir"))
            *md = "vir";
        thresh_dens = vir_density(scale_now) * cons;
    }
    particle_rvir_dens_z0 = vir_density(1.0) * cons;
    return thresh_dens;
}

void calc_mass_definition(void) {
    char   *vir = "vir";
    int64_t i;
    particle_thresh_dens[0] = _calc_mass_definition(&MASS_DEFINITION);
    particle_thresh_dens[1] = _calc_mass_definition(&MASS_DEFINITION2);
    particle_thresh_dens[2] = _calc_mass_definition(&MASS_DEFINITION3);
    particle_thresh_dens[3] = _calc_mass_definition(&MASS_DEFINITION4);
    particle_thresh_dens[4] = _calc_mass_definition(&MASS_DEFINITION5);
    particle_rvir_dens      = _calc_mass_definition(&vir);
    dynamical_time = 1.0 / sqrt((4.0 * M_PI * Gc / 3.0) * particle_rvir_dens *
                                PARTICLE_MASS);
    min_dens_index = 0;
    for (i = 1; i < 5; i++)
        if (particle_thresh_dens[i] < particle_thresh_dens[min_dens_index])
            min_dens_index = i;
}

void lightcone_set_scale(float *pos) {
    int64_t i;
    float   ds = 0, dx = 0, z;
    for (i = 0; i < 3; i++) {
        ds = pos[i] - LIGHTCONE_ORIGIN[i];
        dx += ds * ds;
    }
    z         = comoving_distance_h_to_redshift(sqrt(dx));

#if 0
    // SCALE_NOW is a global variable that is not thread-private.
    SCALE_NOW = scale_factor(z);
#else
    // scale_dx is a global variable that is not thread-private.
    scale_dx = scale_factor(z);
#endif

    calc_mass_definition();
}

void add_new_halo(struct HaloInfo *haloinfo) {
    int64_t i;
    if ((haloinfo->num_halos % 1000) == 0) {
        haloinfo->halos = check_realloc(
            haloinfo->halos, sizeof(struct halo) * (haloinfo->num_halos + 1000),
            "Allocating room for halos.");
        haloinfo->extra_info = check_realloc(
            haloinfo->extra_info,
            sizeof(struct extra_halo_info) * (haloinfo->num_halos + 1000),
            "Allocating room for extra halo info.");
        memset(haloinfo->halos + haloinfo->num_halos, 0,
               sizeof(struct halo) * 1000);
        for (i = haloinfo->num_halos; i < haloinfo->num_halos + 1000; i++) {
            haloinfo->extra_info[i].child =
                haloinfo->extra_info[i].next_cochild =
                    haloinfo->extra_info[i].ph =
                        haloinfo->extra_info[i].prev_cochild =
                            haloinfo->extra_info[i].sub_of = -1;
            haloinfo->extra_info[i].max_metric             = 0;
            haloinfo->halos[i].flags |= GROWING_FLAG;
        }
    }
    haloinfo->num_halos++;
}

void add_more_halo_ids(void) {
    num_alloced_halo_ids += 1000;
    halo_ids = check_realloc(halo_ids, sizeof(int64_t) * num_alloced_halo_ids,
                             "Allocating room for halo ids.");
}

void add_more_growing_halos(void) {
    num_alloc_gh += 1000;
    growing_halos =
        check_realloc(growing_halos, sizeof(struct halo *) * num_alloc_gh,
                      "Allocating room for growing halos.");
}

int dist_compare(const void *a, const void *b) {
    float c = ((struct potential *)a)->r2;
    float d = ((struct potential *)b)->r2;
    if (c > d)
        return 1;
    if (c < d)
        return -1;
    return 0;
}

void _reset_potentials(struct halo *base_h, struct halo *h, float *cen,
                       int64_t p_start, int64_t level, int64_t potential_only) {
    int64_t j, k;
    float   dx, r2;
    memset(po + p_start, 0, sizeof(struct potential) * h->num_p);
    for (j = 0; j < h->num_p; j++) {
        r2 = 0;
        for (k = 0; k < 3; k++) {
            dx = copies[h->p_start + j].pos[k] - cen[k];
            r2 += dx * dx;
        }
        po[p_start + j].r2 = r2;
        memcpy(po[p_start + j].pos, copies[h->p_start + j].pos,
               sizeof(float) * 6);
        if (potential_only)
            po[p_start + j].ke = -1;
        if (h == base_h)
            po[p_start + j].flags = 1;
        if (!potential_only && (h->num_p < base_h->num_p * 0.03))
            po[p_start + j].flags = 2;
    }
}

int64_t calc_particle_radii(struct halo *base_h, struct halo *h, float *cen,
                            int64_t p_start, int64_t level,
                            int64_t                potential_only,
                            const struct HaloInfo *haloinfo) {
    int64_t j, total_p = p_start, child, first_child, parent;

    // Break accidental graph loops
    if (level >= num_alloced_halo_ids)
        add_more_halo_ids();
    halo_ids[level] = h - haloinfo->halos;
    for (j = 0; j < level; j++)
        if (halo_ids[j] == halo_ids[level])
            return p_start;

    _reset_potentials(base_h, h, cen, p_start, level, potential_only);

    first_child = child = haloinfo->extra_info[h - haloinfo->halos].child;
    total_p += h->num_p;
    while (child > -1) {
        total_p =
            calc_particle_radii(base_h, haloinfo->halos + child, cen, total_p,
                                level + 1, potential_only, haloinfo);
        child = haloinfo->extra_info[child].next_cochild;
        assert(child != first_child);
    }

    parent = haloinfo->extra_info[h - haloinfo->halos].sub_of;
    if ((h == base_h) && (parent > -1) &&
        (haloinfo->halos[parent].num_child_particles *
             INCLUDE_HOST_POTENTIAL_RATIO <
         h->num_child_particles)) {
        total_p = calc_particle_radii(base_h, haloinfo->halos + parent, cen,
                                      total_p, level + 1, 1, haloinfo);
    }
    return total_p;
}

#include "properties.c"

void _find_subfofs_at_r(struct fof *f, float target_r,
                        struct FOFInfo *fofinfo) {
    int64_t i;
    init_particle_smallfofs(f->num_p, f->particles, fofinfo);
    for (i = 0; i < f->num_p; i++) {
        fast3tree_find_sphere_skip(phasetree, res, f->particles + i, target_r);
        link_particle_to_fof(f->particles + i, res->num_points, res->points,
                             fofinfo);
    }
    build_fullfofs(fofinfo);
}

#define MAX_PARTICLES_TO_SAMPLE 10000
void _find_subfofs_better2(struct fof *f, float thresh,
                           struct FOFInfo *fofinfo) {
    int64_t i, j, num_test = MAX_PARTICLES_TO_SAMPLE;
    float   target_r = 0;
    if (EXACT_LL_CALC)
        num_test = f->num_p;
    norm_sd(f, thresh);
    fast3tree_rebuild(phasetree, f->num_p, f->particles);
    if (num_test > f->num_p)
        num_test = f->num_p;
    // else
    unsigned int rand_seed = f->num_p;

    for (i = 0; i < num_test; i++) {
        if (num_test == f->num_p)
            j = i;
        else {
            j = rand_r(&rand_seed);
            j <<= 31;
            j += rand_r(&rand_seed);
            j %= (f->num_p);
        }
        particle_r[i] = fast3tree_find_next_closest_distance(
            phasetree, res, f->particles[j].pos);
    }
    target_r = find_median_r(particle_r, num_test, thresh, &rand_seed);
    _find_subfofs_at_r(f, target_r, fofinfo);
}

void reassign_halo_particles(int64_t p_start, int64_t p_end,
                             const struct HaloInfo *haloinfo) {
    int64_t last_halo, j;
    partition_sort_particles(p_start, p_end, copies, particle_halos);
    last_halo                          = particle_halos[p_start];
    haloinfo->halos[last_halo].p_start = p_start;
    for (j = p_start + 1; j < p_end; j++) {
        if (particle_halos[j] == last_halo)
            continue;
        haloinfo->halos[last_halo].num_p =
            j - haloinfo->halos[last_halo].p_start;
        last_halo                          = particle_halos[j];
        haloinfo->halos[last_halo].p_start = j;
    }
    haloinfo->halos[last_halo].num_p = j - haloinfo->halos[last_halo].p_start;
}

int could_be_poisson_or_force_res(struct halo *h1, struct halo *h2,
                                  int64_t *is_force_res) {
    float   dx, r = 0, v = 0, mpe, mve; //, vt1, vt2;
    int64_t k;
    *is_force_res = 0;
    if (!h1->min_pos_err || !h1->min_vel_err)
        return 1;
    for (k = 0; k < 3; k++) {
        dx = h1->pos[k] - h2->pos[k];
        r += dx * dx;
        dx = h1->pos[k + 3] - h2->pos[k + 3];
        v += dx * dx;
    }
    mpe = h1->min_pos_err;
    mve = h1->min_vel_err;
    dx  = (r / mpe + v / mve) / 2.0;
    if (!(dx > 100))
        return 1;

    r = sqrt(r);
    v = sqrt(v);
    if ((h1->r + h2->r > r) && (1.5 * (h1->vrms + h2->vrms)) > v &&
        (r < 1.5 * FORCE_RES)) {
        *is_force_res = 1;
        return 1;
    }
    return 0;
}

int64_t _find_biggest_parent(int64_t h_start, int64_t use_temporal_info,
                             int64_t growing, const struct HaloInfo *haloinfo) {
    int64_t i, j, max_i = h_start, num_m1, num_m2;
    float   m1 = -1, max_vmax = 0, dx, ds, min_ds = 0;
    if (growing) {
        assert(num_growing_halos);
        max_i = growing_halos[0] - haloinfo->halos;
    }
    for (i = h_start; i < haloinfo->num_halos; i++) {
        if (growing && !(haloinfo->halos[i].flags & GROWING_FLAG))
            continue;
        if (haloinfo->halos[i].vmax > haloinfo->halos[max_i].vmax)
            max_i = i;
        if (haloinfo->halos[i].vmax == haloinfo->halos[max_i].vmax &&
            haloinfo->halos[i].num_p > haloinfo->halos[max_i].num_p)
            max_i = i;
    }

    max_vmax = haloinfo->halos[max_i].vmax;
    if (use_temporal_info && TEMPORAL_HALO_FINDING && PARALLEL_IO) {
        for (i = h_start; i < haloinfo->num_halos; i++) {
            if (i == max_i)
                continue;
            if (growing && !(haloinfo->halos[i].flags & GROWING_FLAG))
                continue;
            if (max_vmax * 0.6 < haloinfo->halos[i].vmax) {
                for (dx = 0, j = 0; j < 3; j++) {
                    ds = haloinfo->halos[i].pos[j] -
                         haloinfo->halos[max_i].pos[j];
                    dx += ds * ds;
                }
                if (!min_ds || min_ds > dx)
                    min_ds = dx;
            }
        }
        min_ds = sqrt(min_ds) / 3.0;

        for (i = h_start; i < haloinfo->num_halos; i++) {
            if (i == max_i)
                continue;
            if (growing && !(haloinfo->halos[i].flags & GROWING_FLAG))
                continue;
            if (max_vmax * 0.6 < haloinfo->halos[i].vmax) {
                if (m1 < 0)
                    m1 = find_previous_mass(haloinfo->halos + max_i,
                                            copies +
                                                haloinfo->halos[max_i].p_start,
                                            &num_m1, min_ds, haloinfo);
                float m2 = find_previous_mass(
                    haloinfo->halos + i, copies + haloinfo->halos[i].p_start,
                    &num_m2, min_ds, haloinfo);
                if (m1 && m2 &&
                    ((m2 > m1) || ((m2 == m1) && (num_m2 > num_m1)))) {
                    max_i  = i;
                    m1     = m2;
                    num_m1 = num_m2;
                }
            }
        }
    }
    return max_i;
}

void _fix_parents(int64_t h_start, const struct HaloInfo *haloinfo) {
    int64_t i, j, sub_of, num_m1, num_m2;
    float   m1, m2, dx, ds;

    for (i = h_start; i < haloinfo->num_halos; i++) {
        haloinfo->extra_info[i].next_cochild =
            haloinfo->extra_info[i].prev_cochild =
                haloinfo->extra_info[i].child = -1;
        if (haloinfo->extra_info[i].sub_of == i)
            haloinfo->extra_info[i].sub_of = -1;
    }

    if (TEMPORAL_HALO_FINDING && PARALLEL_IO) {
        for (i = h_start; i < haloinfo->num_halos; i++) {
            sub_of = haloinfo->extra_info[i].sub_of;
            if (sub_of == i)
                sub_of = haloinfo->extra_info[i].sub_of = -1;
            if (sub_of > -1 &&
                haloinfo->halos[i].vmax > 0.6 * haloinfo->halos[sub_of].vmax) {
                for (dx = 0, j = 0; j < 3; j++) {
                    ds = haloinfo->halos[i].pos[j] -
                         haloinfo->halos[sub_of].pos[j];
                    dx += ds * ds;
                }
                dx = sqrt(dx) / 3.0;
                m2 = find_previous_mass(haloinfo->halos + i,
                                        copies + haloinfo->halos[i].p_start,
                                        &num_m2, dx, haloinfo);
                m1 =
                    find_previous_mass(haloinfo->halos + sub_of,
                                       copies + haloinfo->halos[sub_of].p_start,
                                       &num_m1, dx, haloinfo);
                if (m1 && m2 &&
                    ((m2 > m1) || ((m2 == m1) && (num_m2 > num_m1)))) {
                    haloinfo->extra_info[i].sub_of =
                        haloinfo->extra_info[sub_of].sub_of;
                    haloinfo->extra_info[sub_of].sub_of = i;
                    if (haloinfo->extra_info[sub_of].sub_of == sub_of)
                        haloinfo->extra_info[sub_of].sub_of = -1;
                    i--;
                }
            }
        }
    }

    for (i = h_start; i < haloinfo->num_halos; i++) {
        haloinfo->extra_info[i].max_metric = 1e10;
        sub_of                             = haloinfo->extra_info[i].sub_of;
        if (sub_of > -1)
            haloinfo->extra_info[i].max_metric =
                _calc_halo_dist(haloinfo->halos + i, haloinfo->halos + sub_of);
        else
            continue;
        int64_t next_child = haloinfo->extra_info[sub_of].child;
        haloinfo->extra_info[i].next_cochild = next_child;
        haloinfo->extra_info[i].prev_cochild = -1;
        haloinfo->extra_info[sub_of].child   = i;
        if (next_child > -1)
            haloinfo->extra_info[next_child].prev_cochild = i;
    }
}

void output_level(int64_t p_start, int64_t p_end, int64_t h_start,
                  int64_t level, const struct HaloInfo *haloinfo) {
    int64_t i;
    char    buffer[1024];
    snprintf(buffer, 1024, "%s/levels_%f", OUTBASE, SCALE_NOW);
    FILE *output = check_fopen(buffer, "a");
    for (i = p_start; i < p_end; i++) {
        fprintf(output,
                "%f %f %f %f %f %f %" PRId64 " %" PRId64 " %" PRId64 "\n",
                copies[i].pos[0], copies[i].pos[1], copies[i].pos[2],
                copies[i].pos[3], copies[i].pos[4], copies[i].pos[5],
                p[copies[i].id].id, particle_halos[i], level);
    }
    fclose(output);

    snprintf(buffer, 1024, "%s/halos_%f.levels", OUTBASE, SCALE_NOW);
    output = check_fopen(buffer, "a");
    for (i = h_start; i < haloinfo->num_halos; i++) {
        fprintf(output,
                "%f %f %f %f %f %f %" PRId64 " %" PRId64 " %f %f %f %f %" PRId64
                " %" PRId64 " %" PRId64 "\n",
                haloinfo->halos[i].pos[0], haloinfo->halos[i].pos[1],
                haloinfo->halos[i].pos[2], haloinfo->halos[i].bulkvel[0],
                haloinfo->halos[i].bulkvel[1], haloinfo->halos[i].bulkvel[2],
                haloinfo->halos[i].num_p,
                haloinfo->halos[i].num_child_particles, haloinfo->halos[i].r,
                haloinfo->halos[i].vrms, sqrt(haloinfo->halos[i].min_pos_err),
                sqrt(haloinfo->halos[i].min_vel_err), i,
                haloinfo->extra_info[i].sub_of, level);
    }
    fclose(output);
}

void _find_subs(struct fof *f, int64_t level, struct FOFInfo *fofinfo,
                struct HaloInfo *haloinfo) {
    int64_t f_start, f_end, h_start, i, j, f_index;
    int64_t p_start, max_i = 0, is_force_res;

    // Find subFOFs
    p_start      = f->particles - copies;
    f_index      = f - subfofs;
    int64_t f_np = f->num_p;
    _find_subfofs_better2(f, FOF_FRACTION, fofinfo);
    f_start = num_subfofs;
    copy_fullfofs(&subfofs, &num_subfofs, &num_alloced_subfofs, fofinfo);
    f_end = num_subfofs;

    h_start = haloinfo->num_halos;
    for (i = f_start; i < f_end; i++)
        if (subfofs[i].num_p > MIN_HALO_PARTICLES && subfofs[i].num_p < f_np)
            _find_subs(subfofs + i, level + 1, fofinfo, haloinfo);

    // Convert particle positions back to normal
    if (level > 0)
        f = subfofs + f_index;
    for (j = 0; j < f->num_p; j++)
        particle_halos[p_start + j] = -1;
    for (i = h_start; i < haloinfo->num_halos; i++)
        for (j = 0; j < haloinfo->halos[i].num_p; j++)
            particle_halos[haloinfo->halos[i].p_start + j] = i;

    for (j = 0; j < f->num_p; j++) {
        struct particle *c = f->particles + j;
        memcpy(c->pos, p[c->id].pos, sizeof(float) * 6);
    }

    if (h_start == haloinfo->num_halos)
        add_new_halo(haloinfo); // New seed halo
    max_i = _find_biggest_parent(h_start, 1, 0, haloinfo);

    num_growing_halos = 1;
    if (num_growing_halos >= num_alloc_gh)
        add_more_growing_halos();
    haloinfo->halos[max_i].flags |= GROWING_FLAG;
    haloinfo->extra_info[max_i].sub_of = -1;
    growing_halos[0]                   = haloinfo->halos + max_i;
    for (i = h_start; i < haloinfo->num_halos; i++) {
        if ((i == max_i) || !(haloinfo->halos[i].flags & GROWING_FLAG))
            continue;
        if (could_be_poisson_or_force_res(
                haloinfo->halos + i, haloinfo->halos + max_i, &is_force_res)) {
            haloinfo->halos[i].flags -=
                (haloinfo->halos[i].flags & GROWING_FLAG);
            haloinfo->extra_info[i].sub_of = max_i;
            if (!is_force_res) {
                for (j = 0; j < haloinfo->halos[i].num_p; j++)
                    particle_halos[haloinfo->halos[i].p_start + j] = max_i;
                haloinfo->halos[i].num_p = 0;
            }
            continue;
        }
        if (num_growing_halos >= num_alloc_gh)
            add_more_growing_halos();
        growing_halos[num_growing_halos] = haloinfo->halos + i;
        num_growing_halos++;
    }

    if (num_growing_halos == 1) {
        for (j = 0; j < f->num_p; j++)
            if (particle_halos[p_start + j] < 0)
                particle_halos[p_start + j] = max_i;
    } else {
        build_subtree(growing_halos, num_growing_halos);
        for (j = 0; j < f->num_p; j++) {
            if (particle_halos[p_start + j] < 0) {
                struct halo *h = find_best_halo(copies + p_start + j,
                                                haloinfo->halos + max_i);
                particle_halos[p_start + j] = h - haloinfo->halos;
                while (haloinfo->extra_info[h - haloinfo->halos].sub_of > -1) {
                    float max_metric =
                        haloinfo->extra_info[h - haloinfo->halos].max_metric;
                    if (calc_particle_dist(h, copies + p_start + j) >
                        max_metric) {
                        particle_halos[p_start + j] =
                            haloinfo->extra_info[h - haloinfo->halos].sub_of;
                        h = haloinfo->halos +
                            haloinfo->extra_info[h - haloinfo->halos].sub_of;
                    } else
                        break;
                }
            }
        }
    }

    if (TEMPORAL_HALO_FINDING && PARALLEL_IO) {
        for (i = h_start; i < haloinfo->num_halos; i++) {
            if (haloinfo->extra_info[i].ph < 0 ||
                haloinfo->extra_info[i].sub_of < 0)
                continue;
            int64_t sub_of = haloinfo->extra_info[i].sub_of;
            if (haloinfo->extra_info[i].ph == haloinfo->extra_info[sub_of].ph &&
                haloinfo->extra_info[i].max_metric < 1.5) {
                haloinfo->extra_info[i].ph = -1;
                for (j = haloinfo->halos[i].p_start;
                     j < haloinfo->halos[i].num_p + haloinfo->halos[i].p_start;
                     j++)
                    particle_halos[j] = sub_of;
                haloinfo->halos[i].num_p    = haloinfo->halos[i].r =
                    haloinfo->halos[i].vrms = 0;
            }
        }
    }
    reassign_halo_particles(p_start, p_start + f->num_p, haloinfo);
    // calc_num_child_particles(h_start);
    for (i = 0; i < num_growing_halos; i++)
        calc_basic_halo_props(growing_halos[i], haloinfo);
    max_i = _find_biggest_parent(h_start, 0, 1, haloinfo);
    build_subtree(growing_halos, num_growing_halos);
    for (i = 0; i < num_growing_halos; i++)
        haloinfo->extra_info[growing_halos[i] - haloinfo->halos].sub_of =
            find_best_parent(growing_halos[i], haloinfo->halos + max_i) -
            haloinfo->halos;
    _fix_parents(h_start, haloinfo);
    calc_num_child_particles(h_start, haloinfo);
    for (i = 0; i < num_growing_halos; i++)
        calc_basic_halo_props(growing_halos[i], haloinfo);
    if (OUTPUT_LEVELS)
        output_level(p_start, p_start + f->num_p, h_start, level, haloinfo);

    num_subfofs = f_start;
    if (!level) {
        num_growing_halos = haloinfo->num_halos - h_start;
        while (num_alloc_gh < num_growing_halos)
            add_more_growing_halos();
        for (i = 0; i < num_growing_halos; i++)
            growing_halos[i] = haloinfo->halos + h_start + i;
        for (i = 0; i < num_growing_halos; i++) {
            calc_basic_halo_props(growing_halos[i], haloinfo);
            convert_and_sort_core_particles(
                growing_halos[i], copies + growing_halos[i]->p_start, 0, NULL);
        }
        build_subtree(growing_halos, num_growing_halos);
        max_i = _find_biggest_parent(h_start, 0, 0, haloinfo);
        for (i = 0; i < num_growing_halos; i++)
            haloinfo->extra_info[growing_halos[i] - haloinfo->halos].sub_of =
                find_best_parent(growing_halos[i], haloinfo->halos + max_i) -
                haloinfo->halos;
        _fix_parents(h_start, haloinfo);
        calc_num_child_particles(h_start, haloinfo);
    }
}

void find_subs(struct fof *f, struct FOFInfo *fofinfo,
               struct HaloInfo *haloinfo) {
    struct fof cf;
    int64_t    i, h_start = haloinfo->num_halos;

    if (!phasetree)
        phasetree = fast3tree_init(0, NULL);
    if (!res)
        res = fast3tree_results_init();

    if (f->num_p > num_alloc_pc)
        alloc_particle_copies(f->num_p);
    if (!f->num_p)
        return;

    memcpy(copies, f->particles, sizeof(struct particle) * f->num_p);
    for (i = 0; i < f->num_p; i++)
        copies[i].id = (f->particles - p) + i; // Hijack particle IDs
    cf           = *f;
    cf.particles = copies;
    num_copies   = f->num_p;

    if (LIGHTCONE)
        lightcone_set_scale(f->particles->pos);

    num_subfofs = 0;
    _find_subs(&cf, 0, fofinfo, haloinfo);
    num_subfofs = 0;
    for (i = 0; i < f->num_p; i++)
        copies[i] = p[copies[i].id];
    calc_num_child_particles(h_start, haloinfo);
    for (i = h_start; i < haloinfo->num_halos; i++)
        calc_basic_halo_props(haloinfo->halos + i, haloinfo);
    for (i = h_start; i < haloinfo->num_halos; i++)
        calc_additional_halo_props(haloinfo->halos + i, haloinfo);

    memcpy(f->particles, copies, sizeof(struct particle) * f->num_p);
    for (i = h_start; i < haloinfo->num_halos; i++)
        haloinfo->halos[i].p_start += (f->particles - p);
}

void alloc_particle_copies(int64_t total_copies) {
    int64_t max_particle_r = MAX_PARTICLES_TO_SAMPLE;
    if (EXACT_LL_CALC)
        max_particle_r = total_copies;
    if (total_copies - num_alloc_pc < 1000)
        total_copies = num_alloc_pc + 1000;
    check_realloc_s(copies, sizeof(struct particle), total_copies);
    check_realloc_s(particle_halos, sizeof(int64_t), total_copies);
    if (max_particle_r > total_copies)
        max_particle_r = total_copies;
    check_realloc_s(particle_r, sizeof(float), max_particle_r);
    check_realloc_s(po, sizeof(struct potential), total_copies);
    num_alloc_pc = total_copies;
}

void free_particle_copies(void) {
    num_alloc_pc = 0;
    copies       = check_realloc(copies, 0, "Freeing copies.");
    particle_halos =
        check_realloc(particle_halos, 0, "Freeing particle links.");
    particle_r = check_realloc(particle_r, 0, "Freeing particle radii.");
    po         = check_realloc(po, 0, "Freeing potentials.");
    free_subtree();
}

int64_t rad_partition(float *rad, int64_t left, int64_t right,
                      int64_t pivot_ind) {
    float   pivot = rad[pivot_ind], tmp;
    int64_t si, i;
#define SWAP(a, b)                                                             \
    {                                                                          \
        tmp    = rad[a];                                                       \
        rad[a] = rad[b];                                                       \
        rad[b] = tmp;                                                          \
    }
    SWAP(pivot_ind, right - 1);
    si = right - 2;
    for (i = left; i < si; i++) {
        if (rad[i] > pivot) {
            SWAP(i, si);
            si--;
            i--;
        }
    }
    if (rad[si] <= pivot)
        si++;
    SWAP(right - 1, si);
    return si;
#undef SWAP
}

float random_unit(void) {
    return (((float)(rand() % (RAND_MAX)) / (float)(RAND_MAX)));
}

float random_unit_r(unsigned int *seedp) {
    return (((float)(rand_r(seedp) % (RAND_MAX)) / (float)(RAND_MAX)));
}

float find_median_r(float *rad, int64_t num_p, float frac,
                    unsigned int *seedp) {
    int64_t pivot_index, k = num_p * frac;
    int64_t left = 0, right = num_p;
    assert(num_p > 0);
    if (num_p < 2)
        return rad[0];
    while (1) {
        pivot_index = rad_partition(
            rad, left, right, left + random_unit_r(seedp) * (right - left));
        if (k == pivot_index || (rad[left] == rad[right - 1]))
            return rad[k];
        if (k < pivot_index)
            right = pivot_index;
        else
            left = pivot_index + 1;
    }
}

void norm_sd(struct fof *f, float thresh) {
    double  pos[6] = {0};
    double  corr[6][6];
    double  sig_x, sig_v;
    int64_t i, j, k;

    if (!f->num_p)
        return;

    for (i = 0; i < f->num_p; i++)
        for (j = 0; j < 6; j++)
            pos[j] += f->particles[i].pos[j];

    for (j = 0; j < 6; j++)
        pos[j] /= (double)f->num_p;
    for (j = 0; j < 6; j++)
        for (k = 0; k < 6; k++)
            corr[j][k] = 0;

    for (i = 0; i < f->num_p; i++) {
        for (j = 0; j < 6; j++)
            f->particles[i].pos[j] -= pos[j];

        for (j = 0; j < 6; j++)
            for (k = j; k < 6; k++)
                corr[j][k] += f->particles[i].pos[j] * f->particles[i].pos[k];
    }

    for (j = 0; j < 6; j++)
        for (k = j; k < 6; k++)
            corr[j][k] /= (double)f->num_p;

    calc_deviations(corr, &sig_x, &sig_v);
    if (f->num_p == num_copies)
        sig_x *= INITIAL_METRIC_SCALING;
    // else sig_x *= CONTINUED_METRIC_SCALING;

    if (!sig_x || !sig_v)
        return;
    for (i = 0; i < f->num_p; i++)
        for (j = 0; j < 6; j++)
            f->particles[i].pos[j] /= ((j < 3) ? sig_x : sig_v);
}
