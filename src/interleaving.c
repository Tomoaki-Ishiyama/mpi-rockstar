#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include "particle.h"
#include "rockstar.h"
#include "groupies.h"
#include "fof.h"
#include "inthash.h"
#include "config_vars.h"
#include "check_syscalls.h"
#include "bounds.h"
#include <unistd.h>

#define FAST3TREE_TYPE struct bparticle
#include "fast3tree.c"

struct fast3tree         *bp_tree = NULL;
struct fast3tree_results *bp_res  = NULL;
struct bgroup            *bg      = NULL;
int64_t                   num_bg  = 0;
int64_t                   max_gid, our_chunk;
struct inthash           *bg_ih      = NULL;
int64_t                   num_new_bp = 0;

struct bgroup *final_bg       = NULL;
int64_t        num_bg_sets    = 0;
int64_t       *bg_set_sizes   = NULL;
int64_t       *bg_set_indices = NULL;
int64_t       *num_new_groups = NULL;

void set_bp_chunk(int64_t chunk) {
    our_chunk = chunk;
    for (int64_t i = 0; i < num_bp; i++)
        bp[i].chunk = chunk;
}

void clear_bp_data(void) {
    fast3tree_results_free(bp_res);
    bp_res = NULL;
    fast3tree_free(&bp_tree);
    bp     = check_realloc(bp, 0, "Freeing bp");
    num_bp = 0;
}

void clear_bg_data(void) {
    if (bg_ih)
        free_inthash(bg_ih);
    bg_ih  = NULL;
    bg     = check_realloc(bg, 0, "freeing BG");
    num_bg = 0;
}

void clear_final_bg_data(void) {
    final_bg       = check_realloc(final_bg, 0, "freeing final bg");
    bg_set_sizes   = check_realloc(bg_set_sizes, 0, "freeing bg set sizes");
    bg_set_indices = check_realloc(bg_set_indices, 0, "freeing bg set indices");
    num_bg_sets    = 0;
}

int64_t add_new_bgroup(struct bparticle *tbp) {
    if (!(num_bg % 1000))
        bg = check_realloc(bg, sizeof(struct bgroup) * (num_bg + 1000), "BG");
    bg[num_bg].chunk  = tbp->chunk;
    bg[num_bg].tagged = -1;
    bg[num_bg].next   = -1;
    bg[num_bg].head   = num_bg;
    bg[num_bg].id     = tbp->bgid;
    if (tbp->chunk == our_chunk) {
        bg[num_bg].num_p = all_fofs[tbp->bgid].num_p;
        assert(bg[num_bg].num_p);
    } else
        bg[num_bg].num_p = 0;
    num_bg++;
    return (num_bg - 1);
}

int64_t find_bgroup(struct bparticle *tbp) {
    int64_t uid = tbp->chunk * max_gid + tbp->bgid;
    int64_t gid = (int64_t)ih_getval(bg_ih, uid);
    gid--;
    if (gid > -1) {
        if (!(tbp->chunk == bg[gid].chunk && tbp->bgid == bg[gid].id)) {
            fprintf(stderr,
                    "Huh? Asked for chunk %" PRId64 " and id %" PRId64
                    ", and got %" PRId64 ", %" PRId64 " with %" PRId64
                    " particles\n",
                    tbp->chunk, tbp->bgid, bg[gid].chunk, bg[gid].id,
                    bg[gid].num_p);
            assert(0);
        }
    }
    return (gid);
}

struct bgroup *find_bgroup_from_id(int64_t id, int64_t chunk) {
    if (id >= max_gid)
        return NULL;
    int64_t uid = chunk * max_gid + id;
    int64_t gid = (int64_t)ih_getval(bg_ih, uid);
    if (!gid)
        return NULL;
    gid--;
    if (!(chunk == bg[gid].chunk && id == bg[gid].id)) {
        fprintf(stderr,
                "Huh? Asked for chunk %" PRId64 " and id %" PRId64
                ", and got %" PRId64 ", %" PRId64 " (uid=%" PRId64
                ") with %" PRId64 " particles (max_gid=%" PRId64
                "; uid=%" PRId64 "; num_elems=%" PRId64 ")\n",
                chunk, id, bg[gid].chunk, bg[gid].id,
                bg[gid].chunk * max_gid + bg[gid].id, bg[gid].num_p, max_gid,
                uid, bg_ih->elems);
        fprintf(stderr, "PID: %d\n", getpid());
        sleep(100);
        assert(0);
    }
    return (bg + gid);
}

void verify_bgroup_hash(void) {
    int64_t i;
    for (i = 0; i < bg_ih->num_buckets; i++) {
        if (bg_ih->buckets[i].key != IH_INVALID) {
            struct bgroup *tbg = bg + ((int64_t)bg_ih->buckets[i].data) - 1;
            if ((tbg->chunk * max_gid + tbg->id) != bg_ih->buckets[i].key) {
                fprintf(stderr, "Error in bucket %" PRId64 "!\n", i);
            }
        }
    }
}

/* Guarantee that the head particle has the lowest chunk number */
void link_bgroups(int64_t gid1, int64_t gid2) {
    if (gid1 == gid2)
        return;
    if (bg[gid1].head == bg[gid2].head)
        return;
    if (bg[bg[gid1].head].chunk > bg[bg[gid2].head].chunk) {
        int64_t temp = gid2;
        gid2         = gid1;
        gid1         = temp;
    }
    gid2         = bg[gid2].head;
    int64_t tail = gid1;
    while (bg[tail].next > -1) {
        tail = bg[tail].next;
    }
    bg[tail].next = gid2;
    tail          = gid2;
    while (tail > -1) {
        bg[tail].head = bg[gid1].head;
        tail          = bg[tail].next;
    }
}

void mark_bgroup_tree(void) {
    int64_t            i, j;
    struct tree3_node *nodes = bp_tree->root;
    for (i = 0; i < bp_tree->num_nodes; i++) {
        if (nodes[i].flags & FAST3TREE_MARKED)
            continue;
        for (j = 1; j < nodes[i].num_points; j++)
            if (nodes[i].points[j].bgid != nodes[i].points[0].bgid ||
                nodes[i].points[j].chunk != nodes[i].points[0].chunk)
                break;
        if (j == nodes[i].num_points)
            _fast3tree_mark_node(nodes + i, FAST3TREE_MARKED);
    }
}

void build_bgroup_links(void) {
    int64_t i, j;
    bp_tree = fast3tree_init(num_new_bp, bp + (num_bp - num_new_bp));
    mark_bgroup_tree();
    bp_res  = fast3tree_results_init();
    bg_ih   = new_inthash();
    max_gid = -1;
    for (i = 0; i < num_bp; i++) {
        if (bp[i].bgid > max_gid)
            max_gid = bp[i].bgid;
    }
    max_gid++;
    for (i = 0; i < num_bp; i++) {
        int64_t uid = bp[i].chunk * max_gid + bp[i].bgid;
        if (!ih_getval(bg_ih, uid)) {
            int64_t gid = add_new_bgroup(bp + i);
            ih_setval(bg_ih, uid, (void *)(gid + 1));
        }
    }

    float r = AVG_PARTICLE_SPACING * FOF_LINKING_LENGTH;
    if (PERIODIC)
        _fast3tree_set_minmax(bp_tree, 0, BOX_SIZE);
    for (i = 0; i < (num_bp - num_new_bp); i++) {
        int64_t gid1 = find_bgroup(bp + i);
        fast3tree_find_sphere_marked(bp_tree, bp_res, bp[i].pos, r, PERIODIC,
                                     1);
        // if (!PERIODIC) fast3tree_find_sphere(bp_tree, bp_res, bp[i].pos, r);
        // else fast3tree_find_sphere_periodic(bp_tree, bp_res, bp[i].pos, r);

        for (j = 0; j < bp_res->num_points; j++) {
            int64_t gid2 = find_bgroup(bp_res->points[j]);
            assert(gid1 > -1 && gid2 > -1);
            link_bgroups(gid1, gid2);
        }
    }
}

void check_bgroup_sanity(int64_t num_sets, int64_t *set_sizes,
                         struct bgroup *groups) {
    int64_t i, j = 0, l = 0;
    for (i = 0; i < num_sets; i++) {
        l = j + set_sizes[i];
        for (; j < l; j++)
            if (groups[j].next == -1)
                break;
        if (j == l) {
            fprintf(stderr, "[Rockstar error] Bgroup sanity test failed!\n");
            assert(0);
        }
        j = l;
    }
}

/* Fn to return list of linked bgroups given input list */
/* Ignores bgroups if any linked groups belong to a lower chunk number. */
void find_bgroup_sets(int64_t chunk, int64_t *num_sets, int64_t **set_sizes,
                      struct bgroup **groups, int64_t *total_groups) {
    int64_t  i, j, loc = 0, new_total_size = 0;
    int64_t *new_set_sizes =
        check_realloc(NULL, (*num_sets) * sizeof(int64_t), "New set sizes");
    int64_t *set_new_index =
        check_realloc(NULL, (*num_sets) * sizeof(int64_t), "Num in set");
    int64_t *num_new_bgroups =
        check_realloc(NULL, (*num_sets) * sizeof(int64_t), "Num in set");

    // Step 1: Link together groups which are in the same request set
    for (i = 0; i < *num_sets; i++) {
        new_set_sizes[i] = set_new_index[i] = num_new_bgroups[i] = 0;
        struct bgroup *g1                                        = NULL;
        assert(set_sizes[0][i] > 0);
        for (j = loc; j < loc + set_sizes[0][i]; j++) {
            g1 = find_bgroup_from_id(groups[0][j].id, groups[0][j].chunk);
            if (g1) {
                if (groups[0][j].num_p) {
                    assert(!(g1->num_p) || (g1->num_p == groups[0][j].num_p));
                    g1->num_p = groups[0][j].num_p;
                }
                struct bgroup temp  = groups[0][j];
                groups[0][j]        = groups[0][loc];
                groups[0][loc]      = temp;
                bg[g1->head].tagged = -1;
                j++;
                break;
            } else {
                num_new_bgroups[i]++;
            }
        }

        for (; j < loc + set_sizes[0][i]; j++) {
            struct bgroup *g2 =
                find_bgroup_from_id(groups[0][j].id, groups[0][j].chunk);
            if (g2) {
                bg[g2->head].tagged = -1;
                link_bgroups(g1 - bg, g2 - bg);
                if (groups[0][j].num_p) {
                    assert(!(g2->num_p) || (g2->num_p == groups[0][j].num_p));
                    g2->num_p = groups[0][j].num_p;
                }
            } else {
                num_new_bgroups[i]++;
            }
        }
        loc = j;
    }

    // Step 2a: tag group heads to check for group sets which are linked
    // together
    j = 0;
    for (i = 0; i < *num_sets; i++) {
        struct bgroup *g1 =
            find_bgroup_from_id(groups[0][j].id, groups[0][j].chunk);
        j += set_sizes[0][i];

        if (!g1) {
            assert(num_new_bgroups[i] == set_sizes[0][i]);
            new_set_sizes[i] = num_new_bgroups[i];
            continue;
        }

        if (bg[g1->head].chunk < chunk) {
            int64_t gid = g1->head;
            while (gid > -1) {
                if (bg[gid].chunk != our_chunk)
                    bg[gid].num_p = 0;
                gid = bg[gid].next;
            }
            continue;
        }

        if (bg[g1->head].tagged < 0) {
            bg[g1->head].tagged = i;
            new_set_sizes[i] += num_new_bgroups[i];
            int64_t gid = g1->head;
            while (gid > -1) {
                new_set_sizes[i]++;
                gid = bg[gid].next;
            }
        } else {
            new_set_sizes[bg[g1->head].tagged] += num_new_bgroups[i];
        }
    }

    // Step 2b: build new group list
    new_total_size = j = loc = 0;
    for (i = 0; i < *num_sets; i++) {
        set_new_index[i] = new_total_size;
        new_total_size += new_set_sizes[i];
    }
    struct bgroup *new_bgroups = check_realloc(
        NULL, new_total_size * sizeof(struct bgroup), "New bgroups");
    *total_groups = new_total_size;
    for (i = 0; i < *num_sets; i++) {
        struct bgroup *g1 =
            find_bgroup_from_id(groups[0][j].id, groups[0][j].chunk);
        if (!g1) { // Copy all groups over
            for (int64_t k = 0; k < set_sizes[0][i]; k++)
                new_bgroups[set_new_index[i] + k] = groups[0][j + k];
            j += set_sizes[0][i];
            continue;
        }
        if (bg[g1->head].chunk < chunk) {
            j += set_sizes[0][i];
            continue;
        }

        int64_t tagged = bg[g1->head].tagged;
        assert(tagged >= 0 && tagged < *num_sets);
        if (tagged == i) { // Copy all linked groups over
            int64_t gid = g1->head;
            while (gid > -1) {
                new_bgroups[set_new_index[i]] = bg[gid];
                if (bg[gid].chunk != our_chunk)
                    bg[gid].num_p = 0;
                set_new_index[i]++;
                gid = bg[gid].next;
            }
        }
        // Copy all nonlinked groups over
        loc = j;
        for (; j < loc + set_sizes[0][i]; j++) {
            struct bgroup *g2 =
                find_bgroup_from_id(groups[0][j].id, groups[0][j].chunk);
            if (!g2) {
                new_bgroups[set_new_index[tagged]] = groups[0][j];
                set_new_index[tagged]++;
            }
        }
    }

    // Step 3: Collapse set sizes
    for (i = 0, j = 0; j < *num_sets; j++) {
        if (!new_set_sizes[j])
            continue;
        new_set_sizes[i] = new_set_sizes[j];
        i++;
    }
    *num_sets = i;
    free(*groups);
    *groups = new_bgroups;
    free(*set_sizes);
    *set_sizes = new_set_sizes;
    free(set_new_index);
    free(num_new_bgroups);
    check_bgroup_sanity(*num_sets, *set_sizes, *groups);
}

void bgroups_to_setlist(void) {
    int64_t i, total_num_groups = 0, set_num = 0, j = 0;

    // Count number of independent sets, ignoring small unconnected groups
    num_bg_sets      = 0;
    total_num_groups = num_bg;
    for (i = 0; i < num_bg; i++) {
        if (bg[i].head != i)
            continue;
        if (bg[i].next == -1 && bg[i].num_p < MIN_HALO_PARTICLES) {
            total_num_groups--;
            continue;
        }
        if (bg[i].chunk < our_chunk) {
            j = i;
            while (j > -1) {
                total_num_groups--;
                j = bg[j].next;
            }
            continue;
        }
        num_bg_sets++;
    }

    bg_set_sizes = check_realloc(bg_set_sizes, sizeof(int64_t) * (num_bg_sets),
                                 "Allocating set sizes.");
    final_bg =
        check_realloc(final_bg, sizeof(struct bgroup) * (total_num_groups),
                      "Allocating bgroup lists.");

    // Fill group list and set sizes
    j = 0;
    memset(bg_set_sizes, 0, sizeof(int64_t) * num_bg_sets);
    for (i = 0; i < num_bg; i++) {
        if (bg[i].head != i)
            continue;
        if (bg[i].next == -1 && bg[i].num_p < MIN_HALO_PARTICLES)
            continue;
        if (bg[i].chunk < our_chunk)
            continue;
        int64_t gid = i;
        while (gid > -1) {
            final_bg[j] = bg[gid];
            bg_set_sizes[set_num]++;
            gid = bg[gid].next;
            j++;
        }
        set_num++;
    }

    // clear_bg_data();
}

int64_t prune_setlist(void) {
    int64_t i, j = 0, k = 0, loc, num_p_in_set, set_num = 0;
    for (i = 0; i < num_bg_sets; i++) {
        loc          = j;
        num_p_in_set = 0;
        for (; j < loc + bg_set_sizes[i]; j++)
            num_p_in_set += final_bg[j].num_p;
        if (num_p_in_set < MIN_HALO_PARTICLES)
            continue;
        bg_set_sizes[set_num] = bg_set_sizes[i];
        set_num++;
        j = loc;
        for (; j < loc + bg_set_sizes[i]; j++, k++)
            final_bg[k] = final_bg[j];
    }
    return set_num;
}

int64_t calc_next_bgroup_chunk(void) {
    int64_t i, total_bg = 0, max_chunk = 0, total_num_p = 0;
    if (!num_new_groups)
        num_new_groups = check_realloc(NULL, sizeof(int64_t) * NUM_WRITERS,
                                       "Allocating new group counts.");
    memset(num_new_groups, 0, sizeof(int64_t) * NUM_WRITERS);
    for (i = 0; i < num_bg_sets; i++)
        total_bg += bg_set_sizes[i];
    for (i = 0; i < total_bg; i++) {
        assert(final_bg[i].chunk >= 0 && final_bg[i].chunk < NUM_WRITERS);
        total_num_p += final_bg[i].num_p;
        if (!final_bg[i].num_p)
            num_new_groups[final_bg[i].chunk]++;
    }
    for (i = 1; i < NUM_WRITERS; i++)
        if (num_new_groups[i] > num_new_groups[max_chunk])
            max_chunk = i;
    if (num_new_groups[max_chunk])
        return max_chunk;
    return -1;
}

int64_t tag_halo_as_in_bounds(struct halo *h) {
    int64_t cur_h = h - halos;
    if (h->flags & TAGGED_FLAG)
        return 0;
    h->flags |= TAGGED_FLAG;
    int64_t first_child  = extra_info[cur_h].child;
    int64_t tagged_halos = 1;
    cur_h                = first_child;
    while (cur_h != -1) {
        tagged_halos += tag_halo_as_in_bounds(halos + cur_h);
        cur_h = extra_info[cur_h].next_cochild;
        assert(cur_h != first_child);
    }
    return tagged_halos;
}

void sort_out_halos_for_chunk(int64_t chunk, float *bounds,
                              struct workunit_info *w, struct fof **c_fofs,
                              struct halo            **c_halos,
                              struct extra_halo_info **c_ei,
                              struct particle **c_p, struct fof *fofs) {
    int64_t i, c_num_h = 0, c_num_fofs = 0, c_num_p = 0, c_num_meta_p = 0,
               c_loc, c_meta_loc, cur_h, c_num_meta_fofs = 0;

    memcpy(w->bounds, bounds, sizeof(float) * 6);
    for (i = 0; i < num_halos; i++)
        halos[i].flags -= (halos[i].flags & TAGGED_FLAG);
    for (i = 0; i < num_halos; i++) {
        if (halos[i].flags & TAGGED_FLAG)
            continue;
        if (_check_bounds_raw(halos[i].pos, bounds))
            c_num_h += tag_halo_as_in_bounds(halos + i);
    }

    for (i = 0; i < num_halos; i++)
        if (halos[i].flags & TAGGED_FLAG) {
            if (halos[i].p_start >= w->num_particles)
                c_num_meta_p += halos[i].num_p;
            else
                c_num_p += halos[i].num_p;
        }

    if ((c_num_h == num_halos) || (c_num_p > 0.5 * num_p)) {
        *c_fofs  = fofs;
        *c_ei    = extra_info;
        *c_p     = p;
        *c_halos = halos;
        return;
    }

    w->num_halos              = c_num_h;
    int64_t *index_conversion = check_realloc(NULL, num_halos * sizeof(int64_t),
                                              "Halo index conversion");
    *c_p =
        check_realloc(NULL, sizeof(struct particle) * (c_num_p + c_num_meta_p),
                      "chunk particles");
    *c_halos =
        check_realloc(NULL, sizeof(struct halo) * (c_num_h), "chunk halos");
    *c_ei = check_realloc(NULL, sizeof(struct extra_halo_info) * (c_num_h),
                          "chunk info");

    int64_t j = 0, processed_parts = 0, processed_meta_parts = w->num_particles;
    w->num_particles = c_meta_loc = c_num_p;
    w->num_meta_p                 = c_num_meta_p;
    cur_h = c_loc = 0;
    for (i = 0; i < num_halos; i++)
        index_conversion[i] = -1;
    for (i = 0; i < w->num_fofs; i++) {
        int64_t  num_p_from_fof = 0;
        int64_t  num_h_from_fof = 0;
        int64_t *p_start        = &processed_parts;
        int64_t *copy_loc       = &c_loc;
        if (!fofs[i].particles) {
            p_start  = &processed_meta_parts;
            copy_loc = &c_meta_loc;
        }
        int64_t p_end = *p_start + fofs[i].num_p;
        for (; j < num_halos && (halos[j].p_start + halos[j].num_p <= p_end) &&
               (halos[j].p_start >= *p_start);
             j++) {
            if ((halos[j].flags & TAGGED_FLAG)) {
                if (halos[j].num_p == 0 && halos[j].p_start == p_end) {
                    int64_t sub_of = extra_info[j].sub_of;
                    while (sub_of > -1 && extra_info[sub_of].sub_of > -1)
                        sub_of = extra_info[sub_of].sub_of;
                    if (sub_of > -1 &&
                        halos[sub_of].p_start + halos[sub_of].num_p > p_end)
                        break;
                }
                c_halos[0][cur_h]         = halos[j];
                c_halos[0][cur_h].p_start = *copy_loc;
                c_ei[0][cur_h]            = extra_info[j];
                index_conversion[j]       = cur_h;
                cur_h++;
                num_h_from_fof++;
                if (halos[j].num_p) {
                    num_p_from_fof += halos[j].num_p;
                    memcpy((*c_p) + *copy_loc, p + halos[j].p_start,
                           halos[j].num_p * sizeof(struct particle));
                    copy_loc[0] += halos[j].num_p;
                }
            }
        }
        p_start[0] += fofs[i].num_p;
        if (num_h_from_fof && !(num_p_from_fof))
            cur_h -= num_h_from_fof;
        if (num_p_from_fof) {
            if (!(c_num_fofs % 1000))
                *c_fofs = check_realloc(
                    *c_fofs, sizeof(struct fof) * (c_num_fofs + 1000),
                    "chunk fofs");
            c_fofs[0][c_num_fofs] = fofs[i];
            assert((!fofs[i].particles) || (fofs[i].num_p == num_p_from_fof));
            c_fofs[0][c_num_fofs].num_p = num_p_from_fof;
            c_num_fofs++;
            if (!fofs[i].particles)
                c_num_meta_fofs++;
        }
    }
    w->num_halos = c_num_h = cur_h;

#define IC(x) (x = ((x > -1) && (x < num_halos)) ? index_conversion[x] : -1)
    for (i = 0; i < c_num_h; i++) {
        IC(c_ei[0][i].sub_of); // Note that parent may not exist for this chunk
        IC(c_ei[0][i].child);
        IC(c_ei[0][i].next_cochild);
        IC(c_ei[0][i].prev_cochild);
    }
#undef IC
    w->num_fofs      = c_num_fofs;
    w->num_meta_fofs = c_num_meta_fofs;
    w->num_meta_p    = c_num_meta_p;
    w->num_particles = c_num_p;
    w->total_bg      = 0;
    free(index_conversion);
}
