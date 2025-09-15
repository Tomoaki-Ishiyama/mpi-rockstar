#include "error.h"

void factor_3(int64_t in, int64_t *factors) {
    int64_t i, n = 0;
    for (i = ceil(fabs(cbrt(in))); i > 0 && n < 2; i--)
        if ((in % i) == 0) {
            in /= i;
            factors[n++] = i;
            i            = ceil(sqrt(llabs(in))) + 1;
        }
    factors[2] = in;
}

// Divide box into equal numbers of particles
void divide_projection(struct projection *proj, int64_t pieces, float *places) {
    int64_t i, np = 0, n = 1, cp = 0, cpp;
    double  f;

    assert(pieces > 0);
    for (i = 0; i < PROJECTION_SIZE; i++)
        np += proj->data[i];
    np        = (np + pieces - 1) / pieces; // Round up, if possible
    places[0] = 0;
    for (i = 0; i < PROJECTION_SIZE && n < pieces; i++) {
        cp += proj->data[i];
        if (cp > n * np) {
            cpp =
                cp - proj->data[i]; // Calculate fractional location of division
            f = (cp > cpp) ? ((double)((n * np) - cpp) / (double)(cp - cpp))
                           : 0;
            places[n] = BOX_SIZE * (((double)i + f) / (double)PROJECTION_SIZE);
            n++;
        }
    }
    if (n < pieces) {
        print_time();
        fprintf(stderr, "[Rockstar warning] Projection failed; reverting to equal "
                        "volume divisions.\n");
        for (; n < pieces; n++)
            places[n] = places[n - 1] +
                        (BOX_SIZE - places[n - 1]) / (double)(pieces - n + 1);
    }
}

void sort_chunks(void) {
    int64_t tmp;
#define SWAP(a, b)                                                             \
    {                                                                          \
        tmp       = chunks[a];                                                 \
        chunks[a] = chunks[b];                                                 \
        chunks[b] = tmp;                                                       \
    }
    if (chunks[0] > chunks[1])
        SWAP(0, 1);
    if (chunks[0] > chunks[2])
        SWAP(0, 2);
    if (chunks[1] > chunks[2])
        SWAP(1, 2);
#undef SWAP
}

void populate_bounds(int64_t pos, float *oldbounds, float *newbounds,
                     float new_min, float new_max) {
    for (int64_t i = 0; i < 6; i++) {
        if ((i % 3) < pos)
            newbounds[i] = oldbounds[i];
        else if ((i % 3) == pos)
            newbounds[i] = (i < 3) ? new_min : new_max;
        else
            newbounds[i] = (i < 3) ? 0 : BOX_SIZE;
    }
}

void send_projection_requests(struct projection *pr, int64_t num_requests) {
    int64_t                    i, j, num_to_send;
    float                      bounds[6];
    struct projection_request *prq = NULL;
    prq = check_realloc(prq, sizeof(struct projection_request) * num_requests,
                        "Allocating projection requests.");
    for (i = 0; i < NUM_READERS; i++) {
        num_to_send = 0;
        for (j = 0; j < num_requests; j++)
            if (bounds_overlap(clients[i].bounds, pr[j].bounds, bounds, 0)) {
                prq[num_to_send].dir = pr[j].dir;
                prq[num_to_send].id  = pr[j].id;
                memcpy(prq[num_to_send].bounds, pr[j].bounds,
                       sizeof(float) * 6);
                num_to_send++;
            }
        send_to_socket_noconfirm(clients[i].cs, "proj", 4);
        send_to_socket_noconfirm(clients[i].cs, &num_to_send, sizeof(int64_t));
        if (num_to_send)
            send_to_socket_noconfirm(clients[i].cs, prq,
                                     sizeof(struct projection_request) *
                                         num_to_send);
    }
    free(prq);
}

void accumulate_projections(struct projection *pr, int64_t start,
                            int64_t num_requests) {
    int64_t           max_requests = 1 + (1 + chunks[1]) * chunks[0];
    int64_t           i, j, k, num_to_recv;
    char              cmd[5] = {0};
    struct projection cproj;
    for (j = start; j < start + num_requests; j++)
        for (k = 0; k < PROJECTION_SIZE; k++)
            pr[j].data[k] = 0;

    for (i = 0; i < NUM_READERS; i++) {
        recv_from_socket(clients[i].cs, cmd, 4);
        protocol_check(cmd, "cprj");
        recv_from_socket(clients[i].cs, &num_to_recv, sizeof(int64_t));
        assert((num_to_recv >= 0) && (num_to_recv <= max_requests));
        for (j = 0; j < num_to_recv; j++) {
            recv_from_socket(clients[i].cs, &cproj, sizeof(struct projection));
            assert((cproj.id >= start) && (cproj.id < (start + num_requests)));
            for (k = 0; k < PROJECTION_SIZE; k++)
                pr[cproj.id].data[k] += cproj.data[k];
        }
    }
}

void decide_chunks_for_volume_balance() {
    int64_t i, n, idx[3];
    float   bounds[6];
    for (i = 0; i < 3; i++)
        chunk_size[i] = BOX_SIZE / (float)chunks[i];
    for (n = 0; n < NUM_WRITERS; n++) {
        idx[0] = n % chunks[0];
        idx[1] = ((int64_t)(n / chunks[0])) % chunks[1];
        idx[2] = n / (chunks[0] * chunks[1]);
        for (i = 0; i < 3; i++) {
            bounds[i]     = idx[i] * chunk_size[i];
            bounds[i + 3] = (idx[i] + 1 == chunks[i])
                                ? BOX_SIZE
                                : bounds[i] + chunk_size[i];
        }
        memcpy(clients[NUM_READERS + n].bounds, bounds, sizeof(float) * 6);
        send_to_socket_noconfirm(clients[NUM_READERS + n].cs, "bnds", 4);
        send_to_socket_noconfirm(clients[NUM_READERS + n].cs,
                                 clients[NUM_READERS + n].bounds,
                                 sizeof(float) * 6);
    }
}

void decide_chunks_by_script() {
    FILE   *script;
    int64_t n, i;
    int     port, pid;
    float   bounds[6];
    char    buffer[1024];

    script = check_rw_socket(LOAD_BALANCE_SCRIPT, &pid);
    check_fprintf(script, "%" PRId64 " #Num writers\n", NUM_WRITERS);
    check_fprintf(
        script, "%" PRId64 " %" PRId64 " %" PRId64 " #Recommended divisions\n",
        chunks[0], chunks[1], chunks[2]);
    check_fprintf(script, "%f #Box size (comoving Mpc/h)\n", BOX_SIZE);
    check_fprintf(script, "%f #Current scale factor\n", SCALE_NOW);
    check_fprintf(script, "#Format: ID IP_Address Port\n");
    for (n = 0; n < NUM_WRITERS; n++)
        check_fprintf(script, "%" PRId64 " %s %d\n", n,
                      clients[n + NUM_READERS].address,
                      clients[n + NUM_READERS].port);
    check_fprintf(script, "#Expected Return Format: ID IP_Address Port min_x "
                          "min_y min_z max_x max_y max_z\n");
    fflush(script);
    for (n = 0; n < NUM_WRITERS; n++) {
        check_fgets(buffer, 1024, script);
        if (sscanf(buffer, "%" SCNd64 " %*s %d %f %f %f %f %f %f", &i, &port,
                   bounds, bounds + 1, bounds + 2, bounds + 3, bounds + 4,
                   bounds + 5) < 8 ||
            (i != n) || (port != clients[n + NUM_READERS].port)) {
            fprintf(
                stderr,
                "[Rockstar error] Received invalid format from load balance script!\n");
            fprintf(stderr, "[Rockstar error] Offending line: %s", buffer);
            fprintf(stderr,
                    "[Rockstar error] Expected: %" PRId64
                    " %s %d min_x min_y min_z max_x max_y max_z",
                    n, clients[n + NUM_READERS].address,
                    clients[n + NUM_READERS].port);
            exit(1);
        }
        for (i = 0; i < 6; i++)
            if (bounds[i] < 0 || bounds[i] > BOX_SIZE) {
                fprintf(stderr, "[Rockstar error] Received invalid format from load "
                                "balance script!\n");
                fprintf(stderr, "[Rockstar error] Offending line: %s", buffer);
                fprintf(stderr,
                        "[Rockstar error] Bounds must be within the range 0 to %f\n",
                        BOX_SIZE);
                exit(1);
            }
        memcpy(clients[NUM_READERS + n].bounds, bounds, sizeof(float) * 6);
        send_to_socket_noconfirm(clients[NUM_READERS + n].cs, "bnds", 4);
        send_to_socket_noconfirm(clients[NUM_READERS + n].cs,
                                 clients[NUM_READERS + n].bounds,
                                 sizeof(float) * 6);
    }
    rw_socket_close(script, pid);
}

void decide_chunks_for_memory_balance() {
    struct projection *pr = NULL;
    int64_t            num_proj, proj_start, todo, i, j, dir, offset;
    float             *divisions = NULL, *bnds;

    sort_chunks();
    num_proj = 1 + (1 + chunks[1]) * chunks[0];
    check_realloc_s(pr, sizeof(struct projection), num_proj);
    check_realloc_s(divisions, sizeof(float), (chunks[2] + 1));
    for (i = 0; i < num_proj; i++)
        pr[i].id = i;
    populate_bounds(0, NULL, pr[0].bounds, 0, BOX_SIZE);

    print_time();
    fprintf(stderr, "Sending projection requests...\n");

    todo       = 1;
    proj_start = 0;
    for (dir = 0; dir < 3; dir++) {
        for (i = 0; i < todo; i++)
            pr[proj_start + i].dir = dir;
        send_projection_requests(pr + proj_start, todo);
        accumulate_projections(pr, proj_start, todo);
        for (i = 0; i < todo; i++) {
            divide_projection(pr + proj_start + i, chunks[dir], divisions);
            divisions[chunks[dir]] = BOX_SIZE;
            for (j = 0; j < chunks[dir]; j++) {
                offset = i * chunks[dir] + j;
                bnds   = (dir < 2) ? pr[proj_start + todo + offset].bounds
                                   : clients[NUM_READERS + offset].bounds;
                populate_bounds(dir, pr[proj_start + i].bounds, bnds,
                                divisions[j], divisions[j + 1]);
                if (dir == 2) {
                    send_to_socket_noconfirm(clients[NUM_READERS + offset].cs,
                                             "bnds", 4);
                    send_to_socket_noconfirm(clients[NUM_READERS + offset].cs,
                                             bnds, sizeof(float) * 6);
                }
            }
        }

        proj_start += todo;
        todo *= chunks[dir];
    }

    free(divisions);
    free(pr);
}

void load_balance(void) {
    int64_t i, done = 0, no_more_work = NUM_READERS, id;
    int64_t id_offset = 0, next_assigned = 0, num_finished = 0;
    int64_t sent_all_clear = 0;
    char    cmd[5]         = {0};

    for (i = NUM_READERS; i < num_clients; i++) {
        clients[i].status  = 0;
        clients[i].workers = 1;
    }

    while (done < NUM_WRITERS) {
        clear_rsocket_tags();
        for (i = NUM_READERS; i < num_clients; i++)
            tag_rsocket(clients[i].cs);
        select_rsocket(RSOCKET_READ, 0);
        for (i = NUM_READERS; i < num_clients; i++) {
            if (!check_rsocket_tag(clients[i].cs))
                continue;
            recv_from_socket(clients[i].cs, cmd, 4);

            if (!strcmp(cmd, "hcnt")) {
                assert(clients[i].status == 2);
                recv_from_socket(clients[i].cs, &(clients[i].num_halos),
                                 sizeof(int64_t));
                clients[i].status = 3;
                send_to_socket_noconfirm(clients[i].cs, "outp", 4);
                send_to_socket_noconfirm(clients[i].cs, &id_offset,
                                         sizeof(int64_t));
                id_offset += clients[i].num_halos;
            }

            else if (!strcmp(cmd, "err!")) {
                protocol_check(cmd, "err!");
                return;
            }

            else if (!strcmp(cmd, "done")) {
                assert(clients[i].status != 4);
                clients[i].status = 4;
                done++;
            }

            else if (!strcmp(cmd, "nmwk")) {
                recv_from_socket(clients[i].cs, &id, sizeof(int64_t));
                if (id < 0)
                    id = i;
                clients[id].workers--;
                if (!clients[id].status)
                    clients[id].status = 1;
                if ((clients[id].status == 1) && !(clients[id].workers)) {
                    clients[id].status = 2;
                    num_finished++;
                }
                while (no_more_work < num_clients &&
                       clients[no_more_work].status)
                    no_more_work++;
                while (next_assigned >= NUM_READERS &&
                       clients[next_assigned].status)
                    next_assigned--;
                if (next_assigned < no_more_work)
                    next_assigned = num_clients - 1;
                while (next_assigned >= NUM_READERS &&
                       clients[next_assigned].status)
                    next_assigned--;
                if (next_assigned < no_more_work)
                    send_to_socket_noconfirm(clients[i].cs, "fini", 4);
                else {
                    send_to_socket_noconfirm(clients[i].cs, "work", 4);
                    send_to_socket_noconfirm(clients[i].cs, &next_assigned,
                                             sizeof(int64_t));
                    send_to_socket_noconfirm(
                        clients[i].cs, clients[next_assigned].address,
                        strlen(clients[next_assigned].address) + 1);
                    send_to_socket_noconfirm(
                        clients[i].cs, clients[next_assigned].serv_port,
                        strlen(clients[next_assigned].serv_port) + 1);
                    clients[next_assigned].workers++;
                    next_assigned--;
                }
            }

            else {
                fprintf(stderr,
                        "[Rockstar error] Client protocol error! (%s) (%" PRId64 ")\n",
                        cmd, i - NUM_READERS);
                shutdown_clients();
                exit(0);
            }
        }
        if (num_finished == NUM_WRITERS && !sent_all_clear) {
            sent_all_clear = 1;
            for (i = NUM_READERS; i < num_clients; i++)
                send_to_socket_noconfirm(clients[i].cs, "allc", 4);
        }
    }
}
