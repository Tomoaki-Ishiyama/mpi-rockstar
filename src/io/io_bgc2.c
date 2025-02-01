#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "bgc2.h"
#include "io_bgc2.h"
#include "meta_io.h"
#include "io_util.h"
#include "io_internal.h"
#include "../config_vars.h"
#include "../check_syscalls.h"
#include "../universal_constants.h"
#include "../rockstar.h"
#include "../groupies.h"

char              **bgc2_snapnames = NULL;
int64_t             num_bgc2_snaps = 0;
GROUP_DATA_RMPVMAX *gd             = NULL;
// extern double       particle_thresh_dens[5];

void populate_header(struct bgc2_header *hdr, int64_t id_offset, int64_t snap,
                     int64_t chunk, float *bounds) {
    memset(hdr, 0, sizeof(struct bgc2_header));
    hdr->magic             = BGC_MAGIC;
    hdr->version           = 2;
    hdr->num_files         = (PARALLEL_IO) ? NUM_WRITERS : 1;
    hdr->file_id           = chunk;
    hdr->snapshot          = snap;
    hdr->group_type        = GTYPE_SO;
    hdr->format_part_data  = PDATA_FORMAT_PV;
    hdr->format_group_data = GDATA_FORMAT_RMPVMAX;
    hdr->ngroups           = 0;
    hdr->ngroups_total     = 0;
    hdr->min_group_part    = MIN_HALO_OUTPUT_SIZE;

    hdr->npart          = 0;
    hdr->npart_total    = 0;
    hdr->npart_orig     = num_p;
    hdr->valid_part_ids = (!IGNORE_PARTICLE_IDS) ? 1 : 0;

    hdr->max_npart       = 0;
    hdr->max_npart_total = 0;

    hdr->linkinglength = FOF_LINKING_LENGTH;
    calc_mass_definition();
    hdr->overdensity =
        particle_thresh_dens[0] * PARTICLE_MASS / (Om * CRITICAL_DENSITY);
    hdr->time        = SCALE_NOW;
    hdr->redshift    = (SCALE_NOW > 0) ? (1.0 / (SCALE_NOW)-1.0) : 1e10;
    hdr->Omega0      = Om;
    hdr->OmegaLambda = Ol;
    hdr->box_size    = BOX_SIZE;

    if (bounds)
        for (int64_t i = 0; i < 6; i++)
            hdr->bounds[i] = bounds[i];

    hdr->Hubble0   = h0;
    hdr->GravConst = Gc;
    hdr->part_mass = PARTICLE_MASS;
}

void convert_to_extended_particles(struct extended_particle *ep) {
    int64_t i, j;
    p = (void *)ep;
    for (i = num_p + num_additional_p - 1; i >= 0; i--) {
        memmove(ep[i].pos, p[i].pos, sizeof(float) * 6);
        ep[i].hid = -1;
        ep[i].id  = p[i].id;
    }
    for (i = 0; i < num_halos; i++)
        for (j = 0; j < halos[i].num_p; j++)
            ep[halos[i].p_start + j].hid = i;
}

float square_dist_from_center(struct extended_particle *a, float *c) {
    float   ds = 0, dx = 0;
    int64_t i;
    for (i = 0; i < 3; i++) {
        dx = fabs(a->pos[i] - c[i]);
        if (PERIODIC && (dx > (BOX_SIZE / 2.0)))
            dx = BOX_SIZE - dx;
        ds += dx * dx;
    }
    return ds;
}

#define FAST3TREE_TYPE   struct extended_particle
#define FAST3TREE_PREFIX BGC2
#include "../fast3tree.c"

int64_t                   num_ep2 = 0;
struct extended_particle *ep = NULL, *ep2    = NULL;
struct fast3tree         *ep_tree, *ep_tree2 = NULL;
struct fast3tree_results *ep_res, *ep_res2;
float                     ep_old_minmax[6];

void insertion_sort_extended_particles(int64_t min, int64_t max,
                                       struct extended_particle **particles,
                                       float                     *radii) {
    int64_t                   i, pos;
    float                     r;
    struct extended_particle *tmp_p;
    for (i = min + 1; i < max; i++) {
        r     = radii[i];
        tmp_p = particles[i];
        pos   = i;
        while ((pos > min) && ((r < radii[pos - 1]) ||
                               ((r == radii[pos - 1]) &&
                                (tmp_p->id < particles[pos - 1]->id)))) {
            radii[pos]     = radii[pos - 1];
            particles[pos] = particles[pos - 1];
            pos--;
        }
        radii[pos]     = r;
        particles[pos] = tmp_p;
    }
}

void sort_extended_particles(int64_t min, int64_t max,
                             struct extended_particle **particles,
                             float                     *radii) {
    int64_t                   i, si;
    float                     minpivot, maxpivot, pivot, tmp;
    struct extended_particle *tmp_p;
    if (max - min < 2)
        return;

    maxpivot = minpivot = radii[min];
    for (i = min + 1; i < max; i++) {
        if (radii[i] > maxpivot)
            maxpivot = radii[i];
        if (radii[i] < minpivot)
            minpivot = radii[i];
    }
    if ((minpivot == maxpivot) || ((max - min) < 10)) {
        insertion_sort_extended_particles(min, max, particles, radii);
        return;
    }
    pivot = minpivot + (maxpivot - minpivot) / 2.0;
    if (pivot == maxpivot)
        pivot = minpivot;
    si = max - 1;
#define SWAP(a, b)                                                             \
    {                                                                          \
        tmp_p        = particles[a];                                           \
        particles[a] = particles[b];                                           \
        particles[b] = tmp_p;                                                  \
        tmp          = radii[a];                                               \
        radii[a]     = radii[b];                                               \
        radii[b]     = tmp;                                                    \
    }

    for (i = min; i < si; i++)
        if (radii[i] > pivot) {
            SWAP(i, si);
            si--;
            i--;
        }
    if (i == si && radii[si] <= pivot)
        si++;
#undef SWAP
    sort_extended_particles(min, si, particles, radii);
    sort_extended_particles(si, max, particles, radii);
}

void init_extended_particle_tree(void) {
    ep = check_realloc(
        p, sizeof(struct extended_particle) * (num_p + num_additional_p),
        "Allocating extended particle memory.");
    convert_to_extended_particles(ep);
    ep_tree = fast3tree_init(num_p + num_additional_p, ep);
    ep_res  = fast3tree_results_init();
    ep_res2  = fast3tree_results_init();
    if (PERIODIC && BOX_SIZE > 0) {
        memcpy(ep_old_minmax, ep_tree->root->min, 3 * sizeof(float));
        memcpy(ep_old_minmax + 3, ep_tree->root->max, 3 * sizeof(float));
        _fast3tree_set_minmax(ep_tree, 0, BOX_SIZE);
    }

#if 0
    int64_t fid = ep_tree->points[0].id;
    char cbuf[256];
    sprintf(cbuf, "part-%lld", fid);
    FILE *fout = check_fopen(cbuf, "w");
    for( int i=0; i<ep_tree->num_points; i++){
      fprintf( fout, "%lf\t%lf\t%lf\t%lld\n", 
	       ep_tree->points[i].pos[0], ep_tree->points[i].pos[1],
	       ep_tree->points[i].pos[2], ep_tree->points[i].id);
    }
    fclose(fout);
#endif

}

struct extended_particle **do_sphere_request(float cen[3], float r,
                                             int64_t *num_sp) {
    if (PERIODIC && BOX_SIZE > 0)
        fast3tree_find_sphere_periodic(ep_tree, ep_res, cen, r);
    else
        fast3tree_find_sphere(ep_tree, ep_res, cen, r);
    *num_sp = ep_res->num_points;
    return ep_res->points;
}

void free_extended_particle_tree(void) {
    fast3tree_results_free(ep_res);
    fast3tree_results_free(ep_res2);
    fast3tree_free(&ep_tree);
    if (ep_tree2)
        fast3tree_free(&ep_tree2);
}

int64_t check_bgc2_snap(int64_t snap) {
    int64_t i;
    if (bgc2_snapnames == NULL) {
        if (!strlen(BGC2_SNAPNAMES))
            return 0;
        read_input_names(BGC2_SNAPNAMES, &bgc2_snapnames, &num_bgc2_snaps);
    }

    for (i = 0; i < num_bgc2_snaps; i++)
        if ((snapnames && !strcmp(snapnames[snap], bgc2_snapnames[i])) ||
            (!snapnames && atoi(bgc2_snapnames[i]) == snap))
            break;
    if (i == num_bgc2_snaps)
        return 0;
    return 1;
}

void output_bgc2(int64_t id_offset, int64_t snap, int64_t chunk,
                 float *bounds) {
    char                    *buffer = NULL;
    int64_t                  i, j, k, id = 0, write_bgc2_file = 0;
    FILE                    *output = NULL;
    struct bgc2_header      *hdr    = NULL;
    PARTICLE_DATA_PV        *pd     = NULL;
    struct extended_particle temp_p;
    int64_t                  num_to_print = count_halos_to_print(bounds);
    double                   dens_thresh[5];
    int64_t                  npart          = -1;
    int64_t                  num_particle_r = 0;
    float                   *particle_r     = NULL;

    write_bgc2_file = check_bgc2_snap(snap);

    if (write_bgc2_file) {
        assert(BGC2_HEADER_SIZE == sizeof(struct bgc2_header));
        check_realloc_s(buffer, 1025, 1);
        get_output_filename(buffer, 1024, snap, chunk, "bgc2");
        output = check_fopen(buffer, "w");
        check_realloc_s(buffer, 0, 0);
        check_realloc_s(hdr, sizeof(struct bgc2_header), 1);
        memset(hdr, 0, sizeof(struct bgc2_header));
        populate_header(hdr, id_offset, snap, chunk, bounds);
        hdr->ngroups = num_to_print;
        fwrite_fortran(hdr, BGC2_HEADER_SIZE, 1, output);
        if (!num_to_print) {
            fclose(output);
            free(hdr);
            free(buffer);
            return;
        }

        check_realloc_s(gd, sizeof(GROUP_DATA_RMPVMAX), num_to_print);
        fwrite_fortran(gd, sizeof(GROUP_DATA_RMPVMAX), num_to_print, output);
    }

    for (i = 0; i < 5; i++)
        dens_thresh[i] =
            particle_thresh_dens[i] * (4.0 * M_PI / 3.0) * PARTICLE_MASS;
    if (num_ep2) {
        ep_tree2 = fast3tree_init(num_ep2, ep2);
        if (BOX_SIZE > 0 && PERIODIC)
            _fast3tree_set_minmax(ep_tree2, 0, BOX_SIZE);
    }

    for (i = 0; i < num_halos; i++) {
        if (!_should_print(halos + i, bounds))
            continue;
        halos[i].flags |= ALWAYS_PRINT_FLAG;

        float r = BGC2_R * halos[i].r;
        if (STRICT_SO_MASSES)
            r = BGC2_R * max_halo_radius(halos + i);
        if (num_ep2) {
            if (BOX_SIZE > 0 && PERIODIC)
                fast3tree_find_sphere_periodic(ep_tree2, ep_res, halos[i].pos,
                                               r);
            else
                fast3tree_find_sphere(ep_tree2, ep_res, halos[i].pos, r);
        } else
            ep_res->num_points = 0;

	//_fast3tree_find_sphere(ep_tree->root, ep_res, halos[i].pos, r);

	if (BOX_SIZE > 0 && PERIODIC)
	  fast3tree_find_sphere_periodic(ep_tree, ep_res2, halos[i].pos, r);
	else
	  fast3tree_find_sphere(ep_tree, ep_res2, halos[i].pos, r);

	ep_res->num_allocated_points = ep_res->num_points + ep_res2->num_points;
	check_realloc_s(ep_res->points, sizeof(struct extended_particle),
			ep_res->num_allocated_points);
	/*if(chunk==2) fprintf(stderr, "test2b %d %d %d %d %d\n", chunk, ep_res->num_points, 
	  ep_res2->num_points, ep_res->num_allocated_points, num_ep2);*/
#if 0
	if( ep_res->num_points==0){
	  ep_res = ep_res2; 
	  ep_res2 = NULL;
	}
	else if( ep_res2->num_points!=0){
	  memcpy(ep_res->points + ep_res->num_points, ep_res2->points, 
		 sizeof(struct extended_particle) * ep_res2->num_points);
	}
#else
	for( j=0; j<ep_res2->num_points; j++){
	  ep_res->points[j+ep_res->num_points] = ep_res2->points[j];
	}
#endif
	//if(chunk==2) fprintf(stderr, "test3\n");
	ep_res->num_points += ep_res2->num_points;

        check_realloc_var(particle_r, sizeof(float), num_particle_r,
                          ep_res->num_points);
        for (j = 0; j < ep_res->num_points; j++)
            particle_r[j] =
                square_dist_from_center(ep_res->points[j], halos[i].pos);
        sort_extended_particles(0, ep_res->num_points, ep_res->points,
                                particle_r);

        // Get rid of duplicated points
        if (ep_res->num_points > 1) {
            for (j = 1, k = 1; j < ep_res->num_points; j++) {
                if (ep_res->points[j]->id != ep_res->points[j - 1]->id) {
                    ep_res->points[k] = ep_res->points[j];
                    k++;
                }
            }
            ep_res->num_points = k;
        }

        double total_mass = 0;
        npart             = -1;
        for (j = 0; j < ep_res->num_points; j++) {
            float r =
                sqrt(square_dist_from_center(ep_res->points[j], halos[i].pos));
            if (r < FORCE_RES)
                r = FORCE_RES;
            total_mass += PARTICLE_MASS;
            float cur_dens = total_mass / (r * r * r);
            if (STRICT_SO_MASSES)
                for (k = 1; k < 5; k++)
                    if (cur_dens > dens_thresh[k])
                        halos[i].alt_m[k - 1] = total_mass;
            if (cur_dens > dens_thresh[0]) {
                if (STRICT_SO_MASSES)
                    halos[i].m = total_mass;
                npart = j;
            }
        }

        if (STRICT_SO_MASSES)
            halos[i].r = cbrt(halos[i].m / dens_thresh[0]) * 1e3;

        if (!write_bgc2_file)
            continue;
        j = npart;
        if (j < 0) {
            j                  = 0; // Almost certainly due to FP imprecision
            ep_res->num_points = 1;
            if (ep_res->num_allocated_points < 1) {
                check_realloc_s(ep_res->points,
                                sizeof(struct extended_particle *), 1);
                ep_res->num_allocated_points = 1;
            }
            ep_res->points[0] = &temp_p;
            memcpy(temp_p.pos, halos[i].pos, sizeof(float) * 6);
            temp_p.id  = -1 - id - id_offset;
            temp_p.hid = halos[i].id;
        }

        gd[id].id        = id + id_offset;
        gd[id].parent_id = -1;
        gd[id].npart     = j + 1;
        gd[id].radius =
            cbrt((3.0 / (4.0 * M_PI)) * (j + 1) / particle_thresh_dens[0]);
        gd[id].mass  = (j + 1) * PARTICLE_MASS;
        gd[id].vmax  = halos[i].vmax;
        gd[id].rvmax = halos[i].rvmax;
        for (j = 0; j < 3; j++) {
            gd[id].pos[j] = halos[i].pos[j];
            gd[id].vel[j] = halos[i].pos[j + 3];
        }

        gd[id].npart_self = 0;
        for (j = gd[id].npart - 1; j >= (int64_t)gd[id].npart_self; j--) {
            struct extended_particle *tmp;
            if (ep_res->points[j]->hid == i) {
                tmp = ep_res->points[gd[id].npart_self];
                ep_res->points[gd[id].npart_self] = ep_res->points[j];
                ep_res->points[j]                 = tmp;
                gd[id].npart_self++;
                j++;
            }
        }

        hdr->npart += gd[id].npart;
        if (gd[id].npart > hdr->max_npart) {
            hdr->max_npart = gd[id].npart;
            check_realloc_s(pd, sizeof(PARTICLE_DATA_PV), hdr->max_npart);
        }

        for (j = 0; j < gd[id].npart; j++) {
            assert(j < ep_res->num_points);
            pd[j].part_id = ep_res->points[j]->id;
            for (k = 0; k < 3; k++) {
                pd[j].pos[k] = ep_res->points[j]->pos[k];
                if (PERIODIC) {
                    if ((pd[j].pos[k] - halos[i].pos[k]) > BOX_SIZE / 2.0)
                        pd[j].pos[k] -= BOX_SIZE;
                    else if ((pd[j].pos[k] - halos[i].pos[k]) < -BOX_SIZE / 2.0)
                        pd[j].pos[k] += BOX_SIZE;
                }
                pd[j].vel[k] = ep_res->points[j]->pos[k + 3];
            }
        }

        fwrite_fortran(pd, sizeof(PARTICLE_DATA_PV), gd[id].npart, output);
        id++;
    }
    check_realloc_s(particle_r, 0, 0);

    if (STRICT_SO_MASSES) {
        output_binary(id_offset, snap, chunk, bounds, 0);
        if (!strcasecmp(OUTPUT_FORMAT, "BOTH") ||
            !strcasecmp(OUTPUT_FORMAT, "ASCII"))
            output_ascii(id_offset, snap, chunk, bounds);
    }

    if (write_bgc2_file) {
        rewind(output);
        fwrite_fortran(hdr, BGC2_HEADER_SIZE, 1, output);
        fwrite_fortran(gd, sizeof(GROUP_DATA_RMPVMAX), num_to_print, output);
        fclose(output);

        free(hdr);
        free(pd);
        check_realloc_s(gd, 0, 0);
    }

}

void load_bgc2_groups(char *filename, struct bgc2_header *hdr,
                      GROUP_DATA_RMPVMAX **groups, int64_t *num_groups) {
    FILE   *input;
    int64_t new_group_size;

    assert(sizeof(struct bgc2_header) == BGC2_HEADER_SIZE);
    input = check_fopen(filename, "rb");

    fread_fortran(hdr, BGC2_HEADER_SIZE, 1, input, 0);
    assert(hdr->magic == BGC_MAGIC);
    assert(hdr->version == 2);
    assert(hdr->format_group_data == GDATA_FORMAT_RMPVMAX);

    new_group_size =
        sizeof(GROUP_DATA_RMPVMAX) * ((*num_groups) + hdr->ngroups);
    *groups = check_realloc(*groups, new_group_size, "Allocating groups.");
    fread_fortran((*groups) + (*num_groups), sizeof(GROUP_DATA_RMPVMAX),
                  hdr->ngroups, input, 0);
    *num_groups += hdr->ngroups;
    fclose(input);
}
