#ifndef _IO_PKDGRAV3LCP_H_
#define _IO_PKDGRAV3LCP_H_
#include <stdint.h>
#include "../particle.h"

void load_particles_pkdgrav3lcp(char *filename, struct particle **p, int64_t *num_p);

#endif /* _IO_PKDGRAV3LCP_H_ */
