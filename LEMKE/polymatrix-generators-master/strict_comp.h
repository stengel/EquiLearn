#include "matrix.h"
#include "graph.h"
#include "polymatrix.h"

#ifndef STRICT_COMP_H
#define STRICT_COMP_H

matrix_t **generate_strict_comp(int m, int n);
polymatrix_t *generate_polymatrix_strict_comp(int actions, double p, char *graph, int g1, int g2);

#endif
