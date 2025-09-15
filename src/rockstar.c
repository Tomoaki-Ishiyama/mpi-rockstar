/* The Rockstar Halo Finder.
   Copyright (C) 2011-2013  Peter Behroozi

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
   If so, it should be in a file called "LICENSE".
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <inttypes.h>
#include <unistd.h>
#include "rockstar.h"
#include "particle.h"
#include "fof.h"
#include "groupies.h"
#include "check_syscalls.h"
#include "config_vars.h"
#include "io/meta_io.h"
#include "bitarray.h"

#define FAST3TREE_TYPE   struct particle
#define FAST3TREE_PREFIX ROCKSTAR
#define POINTS_PER_LEAF  20
#include "fast3tree.c"

struct particle          *p          = NULL;
struct particle          *original_p = NULL;
struct bparticle         *bp         = NULL;
int64_t                   num_p = 0, num_bp = 0, num_additional_p = 0;
struct fast3tree         *tree         = NULL;
struct fast3tree_results *rockstar_res = NULL;
char                     *skip         = NULL;
struct fof               *all_fofs     = NULL;
int64_t                   num_all_fofs = 0, num_bfofs = 0, num_metafofs = 0;
int64_t                   num_fofs_tosend = 0;
int64_t                  *fof_order       = NULL;

void rockstar(float *bounds, int64_t manual_subs) {
    int64_t i;
    float   r;
    float   bounds2[6];

    calc_mass_definition();
    r = AVG_PARTICLE_SPACING * FOF_LINKING_LENGTH;
    if (FORCE_RES * SCALE_NOW > FORCE_RES_PHYS_MAX)
        FORCE_RES = FORCE_RES_PHYS_MAX / SCALE_NOW;
    build_particle_tree();
    struct FOFInfo fofinfo;
    init_fofinfo(&fofinfo);
    init_particle_smallfofs(num_p, p, &fofinfo);
    skip = BIT_ALLOC(num_p);
    BIT_ALL_CLEAR(skip, num_p);

    for (i = 0; i < num_p; i++) {
        if (BIT_TST(skip, i))
            continue;
        fast3tree_find_sphere(tree, rockstar_res, p[i].pos, r);
        link_particle_to_fof(p + i, rockstar_res->num_points,
                             rockstar_res->points, &fofinfo);
        if (rockstar_res->num_points > FOF_SKIP_THRESH) {
            for (int64_t j = 0; j < rockstar_res->num_points; j++)
                BIT_SET(skip, (rockstar_res->points[j] - p));
            fast3tree_find_sphere(tree, rockstar_res, p[i].pos, 2.0 * r);
            link_fof_to_fof(p + i, rockstar_res->num_points,
                            rockstar_res->points, &fofinfo);
        }
    }
    skip = check_realloc(skip, 0, "Freeing skip memory.");
    if (bounds) {
        for (i = 0; i < 3; i++) {
            bounds2[i] = bounds[i] +
                         r * 1.01; // Include extra buffer for round-off error.
            bounds2[i + 3] = bounds[i + 3] - r * 1.01;
        }
        fast3tree_find_outside_of_box(tree, rockstar_res, bounds2);
        fast3tree_free(&tree);
        num_bp = rockstar_res->num_points;
        bp     = (struct bparticle *)check_realloc(
            bp, sizeof(struct bparticle) * num_bp, "boundary particles");

        for (i = 0; i < rockstar_res->num_points; i++) {
            bp[i].id = rockstar_res->points[i]->id;
            memcpy(bp[i].pos, rockstar_res->points[i]->pos, sizeof(float) * 6);
            bp[i].bgid =
                tag_boundary_particle(rockstar_res->points[i], &fofinfo);
            bp[i].chunk = 0;
        }
    }

    clear_particle_tree();
    build_fullfofs(&fofinfo);
    all_fofs   = return_fullfofs(&num_all_fofs, &num_bfofs, &fofinfo);
    original_p = p;

    if (bounds)
        for (i = 0; i < num_bp; i++)
            bp[i].bgid += num_all_fofs - num_bfofs;

    if (!manual_subs) {
        struct HaloInfo haloinfo;
        init_haloinfo(&haloinfo);
        for (i = 0; i < num_all_fofs; i++)
            find_subs(all_fofs + i, &fofinfo, &haloinfo);

        realloc_halos(num_halos + haloinfo.num_halos);
        copy_halos(&haloinfo, num_halos);
        num_halos += haloinfo.num_halos;
        free_haloinfo(&haloinfo);

        rockstar_cleanup();
    }
    free_fofinfo(&fofinfo);
}

void convert_bgroups_to_metafofs(void) {
    int64_t i, j = 0, max_j, num_p_in_set;
    bg_set_indices = check_realloc(
        bg_set_indices, sizeof(int64_t) * num_bg_sets, "Allocating bg indices");
    all_fofs     = check_realloc(all_fofs,
                                 sizeof(struct fof) * (num_all_fofs + num_bg_sets),
                                 "Allocating metafofs");
    num_metafofs = num_bg_sets;
    struct fof mf;
    mf.particles = 0;
    for (i = 0; i < num_bg_sets; i++) {
        num_p_in_set      = 0;
        bg_set_indices[i] = j;
        max_j             = j + bg_set_sizes[i];
        for (; j < max_j; j++)
            num_p_in_set += final_bg[j].num_p;
        mf.num_p                   = num_p_in_set;
        all_fofs[num_all_fofs + i] = mf;
    }
    num_all_fofs += num_metafofs;
    fof_order =
        check_realloc(fof_order, sizeof(int64_t) * (num_all_fofs - num_bfofs),
                      "Allocating fof order");
    for (i = 0; i < (num_all_fofs - num_metafofs - num_bfofs); i++)
        fof_order[i] = i;
    for (; i < (num_all_fofs - num_bfofs); i++)
        fof_order[i] = i + num_bfofs;
    qsort(fof_order, num_all_fofs - num_bfofs, sizeof(int64_t), sort_fofs);
    num_fofs_tosend = num_all_fofs - num_bfofs;
}

void find_unfinished_workunit(struct workunit_info *w, struct fof **fofs,
                              struct particle **parts, int64_t **set_sizes,
                              struct bgroup **bgroup_list) {
    int64_t     i, np = 0, mf_id;
    struct fof *tf   = *fofs;
    w->num_particles = w->num_fofs = w->num_halos = 0;
    w->num_meta_fofs = w->total_bg = w->num_meta_p = 0;
    for (;
         num_fofs_tosend > 0 && (w->num_particles + w->num_meta_p < LARGE_FOF);
         num_fofs_tosend--) {
        if (!(w->num_fofs % 1000))
            tf = *fofs =
                check_realloc(*fofs, sizeof(struct fof) * (w->num_fofs + 1000),
                              "Allocating workunit fofs.");
        int64_t fofid_tosend = fof_order[num_fofs_tosend - 1];
        tf[w->num_fofs]      = all_fofs[fofid_tosend];
        if (fofid_tosend < num_all_fofs - num_bfofs)
            w->num_particles += tf[w->num_fofs].num_p;
        else {
            assert(!tf[w->num_fofs].particles);
            w->num_meta_p += tf[w->num_fofs].num_p;
            mf_id = fofid_tosend - (num_all_fofs - num_metafofs);
            if (!(w->num_meta_fofs % 1000))
                *set_sizes = check_realloc(
                    *set_sizes, sizeof(int64_t) * (w->num_meta_fofs + 1000),
                    "Allocating set_sizes.");
            set_sizes[0][w->num_meta_fofs] = bg_set_sizes[mf_id];
            for (i = 0; i < bg_set_sizes[mf_id]; i++) {
                if (!(w->total_bg % 1000))
                    *bgroup_list = check_realloc(*bgroup_list,
                                                 sizeof(struct bgroup) *
                                                     (w->total_bg + 1000),
                                                 "Allocating bgroups.");
                bgroup_list[0][w->total_bg] =
                    final_bg[bg_set_indices[mf_id] + i];
                w->total_bg++;
            }
            w->num_meta_fofs++;
        }
        w->num_fofs++;
    }

    if (w->num_fofs == 1) {
        *parts = check_realloc(*parts, 0, "Freeing workunit particles.\n");
        if (tf[0].particles)
            *parts = p + (tf[0].particles - original_p);
        else
            *parts = NULL;
    } else {
        *parts =
            check_realloc(*parts, sizeof(struct particle) * w->num_particles,
                          "Allocating workunit particles.");
        for (i = 0; i < w->num_fofs; i++) {
            if (!tf[i].particles)
                continue;
            memcpy(*parts + np, p + (tf[i].particles - original_p),
                   sizeof(struct particle) * tf[i].num_p);
            np += tf[i].num_p;
        }
    }
}

void fof_of_id(int64_t id, struct fof *tf) {
    assert(id >= num_all_fofs - num_metafofs - num_bfofs);
    assert(id < num_all_fofs - num_metafofs);
    tf->num_p     = all_fofs[id].num_p;
    tf->particles = p + (all_fofs[id].particles - original_p);
}

/*
void integrate_finished_workunit(struct workunit_info *w, struct fof *fofs,
                                 struct halo *h, struct extra_halo_info *ei,
                                 struct particle *parts) {
    int64_t i, j = 0, offset, np = 0, np2 = w->num_particles,
               hstart          = num_halos, max_np;
    int64_t          num_add_p = 0, ignore_halos = 0;
    struct particle *next_new_p_loc;
    for (i = 0; i < w->num_fofs; i++)
        if (!fofs[i].particles)
            num_add_p += fofs[i].num_p;
    if (num_add_p)
        p = check_realloc(
            p, sizeof(struct particle) * (num_p + num_additional_p + num_add_p),
            "Allocating additional particles");
    next_new_p_loc = p + num_p + num_additional_p;
    num_additional_p += num_add_p;

    for (i = 0; i < w->num_fofs; i++) {
        ignore_halos = 0;
        if (!fofs[i].particles) {
            fofs[i].particles = next_new_p_loc;
            next_new_p_loc += fofs[i].num_p;
            memcpy(fofs[i].particles, parts + np2,
                   sizeof(struct particle) * fofs[i].num_p);
            offset = (fofs[i].particles - p) - np2;
            np2 += fofs[i].num_p;
            max_np = np2;
        } else {
            offset            = fofs[i].particles - original_p;
            fofs[i].particles = p + offset;
            if (w->chunk == our_chunk)
                memcpy(fofs[i].particles, parts + np,
                       sizeof(struct particle) * fofs[i].num_p);
            else
                ignore_halos = 1;
            offset = (fofs[i].particles - p) - np;
            np += fofs[i].num_p;
            max_np = np;
        }
        for (; (j < w->num_halos) && (h[j].p_start + h[j].num_p <= max_np) &&
               (h[j].p_start >= (max_np - fofs[i].num_p));
             j++) {
            if (ignore_halos)
                continue;
            add_new_halo();
            if (ei[j].child >= 0)
                ei[j].child += hstart;
            if (ei[j].next_cochild >= 0)
                ei[j].next_cochild += hstart;
            if (ei[j].prev_cochild >= 0)
                ei[j].prev_cochild += hstart;
            if (ei[j].sub_of >= 0)
                ei[j].sub_of += hstart;
            halos[num_halos - 1] = h[j];
            halos[num_halos - 1].p_start += offset;
            extra_info[num_halos - 1] = ei[j];
        }
    }
}
*/

int sort_fofs(const void *a, const void *b) {
    const struct fof *c = all_fofs + *((int64_t *)a);
    const struct fof *d = all_fofs + *((int64_t *)b);
    if (c->num_p < d->num_p)
        return -1;
    if (c->num_p > d->num_p)
        return 1;
    return 0;
}

void align_particles(struct fof f) {
    int64_t i, j;
    float   bounds[6], max_counts;
    int64_t counts[100];
    if (!BOX_SIZE || f.num_p < 2)
        return;
    for (j = 0; j < 3; j++) {
        bounds[j] = bounds[j + 3] = f.particles[0].pos[j];
    }
    for (i = 1; i < f.num_p; i++) {
        for (j = 0; j < 3; j++) {
            if (bounds[j] > f.particles[i].pos[j])
                bounds[j] = f.particles[i].pos[j];
            if (bounds[j + 3] < f.particles[i].pos[j])
                bounds[j + 3] = f.particles[i].pos[j];
        }
    }
    for (j = 0; j < 3; j++) {
        if (bounds[j] <= AVG_PARTICLE_SPACING * FOF_LINKING_LENGTH &&
            bounds[j + 3] >=
                (BOX_SIZE - AVG_PARTICLE_SPACING * FOF_LINKING_LENGTH)) {
            memset(counts, 0, sizeof(int64_t) * 100);
            max_counts = (int64_t)((
                BOX_SIZE / (AVG_PARTICLE_SPACING * FOF_LINKING_LENGTH)));
            if (max_counts > 100)
                max_counts = 100;
            double multiple = max_counts / BOX_SIZE;
            for (i = 0; i < f.num_p; i++)
                counts[(int64_t)(f.particles[i].pos[j] * multiple)]++;
            for (i = 0; i < max_counts; i++)
                if (!counts[i])
                    break;
            double wrap_position = (i + 0.5) * BOX_SIZE / max_counts;
            for (i = 0; i < f.num_p; i++)
                if (f.particles[i].pos[j] > wrap_position)
                    f.particles[i].pos[j] -= BOX_SIZE;
        }
    }
}

/*
void do_workunit(struct workunit_info *w, struct fof *fofs) {
    int64_t    i, processed_parts = 0, processed_parts2 = w->num_particles;
    struct fof tmp;
    halos     = check_realloc(halos, 0, "Freeing halo memory.");
    num_halos = 0;
    for (i = 0; i < w->num_fofs; i++) {
        tmp = fofs[i];
        if (fofs[i].particles)
            tmp.particles = p + processed_parts;
        else {
            tmp.particles = p + processed_parts2;
            if (PERIODIC)
                align_particles(tmp);
        }
        find_subs(&tmp);
        if (fofs[i].particles)
            processed_parts += tmp.num_p;
        else
            processed_parts2 += tmp.num_p;
    }
    assert((processed_parts == w->num_particles) &&
           (processed_parts2 == (w->num_meta_p + w->num_particles)));
}
*/

void build_particle_tree(void) {
    int64_t          i, dup_particles = 0;
    struct particle *last_p;
    tree         = fast3tree_init(num_p, p);
    rockstar_res = fast3tree_results_init();
    if (IGNORE_PARTICLE_IDS)
        for (i = 0; i < num_p; i++)
            p[i].id = i;

    if (num_p < 2)
        return;
    last_p = p + (num_p - 1);
    for (i = num_p - 2; i >= 0; i--) {
        if (!memcmp(p[i].pos, last_p->pos, sizeof(float) * 6) ||
            last_p->id == p[i].id) {
            num_p--;
            p[i] = p[num_p];
            dup_particles++;
            continue;
        }
        last_p = p + i;
    }
    if (dup_particles) {
        fast3tree_rebuild(tree, num_p, p);
        if (dup_particles > 0.0001 * num_p)
            fprintf(stderr,
                    "[Rockstar warning] %" PRId64 " duplicate particles removed.\n",
                    dup_particles);
    }
}

void clear_particle_tree(void) {
    fast3tree_results_free(rockstar_res);
    rockstar_res = NULL;
    fast3tree_free(&tree);
}

void rockstar_cleanup() {
    check_realloc_s(all_fofs, 0, 0);
    free_particle_copies();
    check_realloc_s(fof_order, 0, 0);
    num_all_fofs = num_metafofs = num_bfofs = 0;
}

void particle_cleanup() {
    p     = check_realloc(p, 0, "Freeing particle memory");
    num_p = num_additional_p = 0;
    original_p               = NULL;
}

struct particle **find_halo_sphere(struct halo *h, int64_t *num_results) {
    fast3tree_find_sphere(tree, rockstar_res, h->pos, h->r * 1.1e-3);
    *num_results = rockstar_res->num_points;
    return rockstar_res->points;
}
