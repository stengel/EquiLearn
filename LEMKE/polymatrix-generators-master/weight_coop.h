#include "graph.h"
#include "polymatrix.h"

#ifndef WEIGHT_COOP
#define WEIGHT_COOP

matrix_t **generate_weight_coop(int m, double n, int *a1, int *a2);
polymatrix_t *generate_polymatrix_weight_coop(int actions, double p, char *graph, int g1, int g2);

#endif
